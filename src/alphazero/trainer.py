"""
Trainer: orchestration class wrapping train / evaluate / save / load
"""

# System package
from pathlib import Path
import dataclasses, time

# GPU package
import jax, jax.numpy as jnp
import equinox as eqx
import optax

# Rich package
from rich.console import Console

# AlphaZero packages
from .config import Meta, ConfigTrain, ConfigEval
from .core.search import make_recurrent_fn, make_search_fn
from .core.metrics import aggregate_eval
from .parallel.mesh import ShardingPlan
from .parallel.stages import make_collect, make_update, make_evaluate
from .display import (
    save_json, load_json, make_progress, status,
    print_iter, print_compiled, print_resumed,
)


_OPT = {"adam": optax.adam, "sgd": optax.sgd, "adamw": optax.adamw}


class TrainerState(eqx.Module):
    network: eqx.Module
    opt_state: optax.OptState
    key_train: jax.Array
    key_eval: jax.Array
    idx_iter: jax.Array


class Trainer:
    """
    Orchestrater
    ToDo: temperature evolution through the game
    """

    def __init__(self, meta, env, network_fn, config_train, config_eval, dir_save, gpu_mesh=None):
        """
        Init the class (self.* values)
        """
        self.meta = meta
        self.env = env
        self.network_fn = network_fn
        self.config_train = config_train
        self.config_eval = config_eval
        self.dir_save = Path(dir_save)
        # Optimizer
        self.optimizer = _OPT[config_train.optimizer](**config_train.optimizer_kwargs)
        # MCTS mechanics (one pair per config: train explores, eval is greedy)
        self.recurrent_fn_train, self.search_fn_train = self._make_mcts(config_train)
        self.recurrent_fn_eval, self.search_fn_eval = self._make_mcts(config_eval)
        # Device mesh and sharded stages
        self._make_gpu(gpu_mesh)
        # Mutable state
        self.trainer_state = None
        self.logs = []

    # ===== BUILD =====

    def _make_mcts(self, config):
        """
        Build the MCTS pair (dynamics + policy) for a given config
        """
        recurrent_fn = make_recurrent_fn(self.env, discount=config.mcts.discount)
        search_fn = make_search_fn(
            kind=config.mcts.kind,
            num_sim=config.mcts.num_sim,
            **config.mcts.policy_kwargs,
        )
        return recurrent_fn, search_fn

    def _make_gpu(self, gpu_mesh=None):
        """
        Set up the sharding plan (validates sizes) and build the three sharded stages
        """
        # Set the plan
        self.plan = ShardingPlan(
            config_train=self.config_train,
            config_eval=self.config_eval,
            mesh=gpu_mesh,
        )
        # Set the mesh
        jax.set_mesh(self.plan.mesh)
        self.meta.n_gpu = self.plan.n_gpu
        # Set jit functions
        self.collect = make_collect(
            plan=self.plan,
            env=self.env,
            recurrent_fn=self.recurrent_fn_train,
            search_fn=self.search_fn_train,
            config_train=self.config_train,
        )
        self.update = make_update(
            plan=self.plan,
            optimizer=self.optimizer,
            config_train=self.config_train,
        )
        self.evaluate = make_evaluate(
            plan=self.plan,
            env=self.env,
            recurrent_fn=self.recurrent_fn_eval,
            search_fn=self.search_fn_eval,
            config_eval=self.config_eval,
        )

    # ===== ENGINE =====

    def _train_iter(self, network, opt_state, key):
        """
        One training iteration: two jitted calls
        """
        key_roll, key_update = jax.random.split(key)

        # --- Collect self-play data ---
        tic = time.perf_counter()
        subkeys_roll = self.plan.split_keys(key_roll)
        data_roll = self.collect({
            "network": network,
            "key": subkeys_roll,
        })
        jax.block_until_ready(data_roll)
        tac_roll = time.perf_counter() - tic

        # --- Update ---
        tic = time.perf_counter()
        subkeys_update = self.plan.split_keys(key_update)
        network, opt_state, logs_nn = self.update({
            "network": network,
            "opt_state": opt_state,
            "data_roll": data_roll,
            "key": subkeys_update,
        })
        jax.block_until_ready(logs_nn)
        tac_update = time.perf_counter() - tic

        # --- Metrics ---
        logs_nn = jax.device_get(logs_nn)
        log = {
            "time_roll": tac_roll,
            "time_updates": tac_update,
            "value_mean": float(data_roll["value"].mean()),
            "logs_nn": {k: v.tolist() for k, v in logs_nn.items()},
            "logs_nn_mean": {k: float(v.mean()) for k, v in logs_nn.items()},
        }
        return network, opt_state, log

    def _evaluate(self, network, key):
        """
        Greedy evaluation: sharded rollout, then metrics on the full batch
        """
        length, score, success = self.evaluate({
            "network": network,
            "key": self.plan.split_keys(key),
        })
        return aggregate_eval(length=length, score=score, success=success)

    # ===== TRAINER STATE =====

    def init(self):
        """
        Init the Trainer state (TrainerState class)
        """
        # Key management
        # NB: manage new jax.random key generation (used old PRNGKey to avoid equinox save conflict)
        key_train = jax.random.PRNGKey(self.config_train.seed)
        key_eval = jax.random.PRNGKey(self.config_eval.seed)
        # Setup fresh network and optimizer
        _, key_net = jax.random.split(key_train)
        network = self.network_fn(self.env, key_net)
        opt_state = self.optimizer.init(eqx.filter(network, eqx.is_array))
        # Replicate the network across devices
        network, opt_state = self.plan.replicate((network, opt_state))
        # Set the initial State
        self.trainer_state = TrainerState(
            network=network,
            opt_state=opt_state,
            key_train=key_train,
            key_eval=key_eval,
            idx_iter=jnp.array(0, dtype=jnp.int32),
        )
        self.logs = []
        return self

    def _check_trainer_state(self):
        """
        Check trainer_state integrity before `fit(self)`
        """
        s = self.trainer_state
        if s is None:
            raise RuntimeError(
                "trainer_state is `None`, please call .init() for a fresh run, or Trainer.resume(...) to continue from a checkpoint"
            )
        missing = [f for f in ("network", "opt_state", "key_train", "key_eval", "idx_iter")
                   if getattr(s, f, None) is None]
        if missing:
            raise RuntimeError(f"trainer_state is incomplete, missing: {', '.join(missing)}")
        idx = int(s.idx_iter)
        if idx >= self.config_train.n_iter:
            raise RuntimeError(
                f"nothing to do: idx_iter={idx} >= n_iter={self.config_train.n_iter} "
                f"please raise n_iter to continue training"
            )

    # ===== DISK SAVE =====

    def _save_meta_configs(self, dir_save):
        """
        meta + configs: written on every save and at the start of every fit
        """
        save_json(dir_save / "meta.json", dataclasses.asdict(self.meta))
        save_json(dir_save / "configs.json", {
            "train": dataclasses.asdict(self.config_train),
            "eval": dataclasses.asdict(self.config_eval),
        })

    def save(self, dir_save=None):
        """
        Post-training save
        Save the metadata, config, networks and logs at the root of `dir_save`
        """
        # Dir management
        dir_save = Path(dir_save) if dir_save is not None else self.dir_save
        dir_save.mkdir(parents=True, exist_ok=True)
        # Save metadata, configs and logs
        self._save_meta_configs(dir_save)
        save_json(dir_save / "logs.json", self.logs)
        # Save the network
        eqx.tree_serialise_leaves(dir_save / "network.eqx", self.trainer_state.network)

    def checkpoint(self, dir_save=None, label=None):
        """
        During-training save
        Save the checkpoint in `{dir_save}/checkpoints/checkpoint_{label}`
        Save the ongoin TrainState and logs
        Any training can be resume from a checkpoint
        """
        # Dir management
        dir_save = Path(dir_save) if dir_save is not None else self.dir_save
        name = label if label is not None else str(int(self.trainer_state.idx_iter))
        dir_checkpoint = dir_save / "checkpoints" / f"checkpoint_{name}"
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        # Save the checkpoint (state and logs)
        eqx.tree_serialise_leaves(dir_checkpoint / "state.eqx", self.trainer_state)
        save_json(dir_checkpoint / "logs.json", self.logs)

    @classmethod
    def _from_disk(cls, dir_save, env, network_fn, gpu_mesh=None):
        """
        Rebuild a fresh trainer from the meta/configs on disk
        """
        meta = Meta(**load_json(dir_save / "meta.json"))
        config = load_json(dir_save / "configs.json")
        self = cls(
            meta=meta,
            env=env,
            network_fn=network_fn,
            config_train=ConfigTrain(**config["train"]),
            config_eval=ConfigEval(**config["eval"]),
            dir_save=dir_save,
            gpu_mesh=gpu_mesh,
        )
        # Builds the state template with the right sharding
        return self.init()

    @classmethod
    def resume(cls, dir_save, env, network_fn, gpu_mesh=None, label="end", verbose=True):
        # Dir management
        dir_save = Path(dir_save)
        dir_checkpoint = dir_save / "checkpoints" / f"checkpoint_{label}"
        if not dir_checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {dir_checkpoint}")

        # Build a fresh trainer from the saved meta/configs
        self = cls._from_disk(dir_save, env, network_fn, gpu_mesh)

        # Get old/new gpu
        n_gpu_old = load_json(dir_save / "meta.json").get("n_gpu")
        n_gpu_new = self.meta.n_gpu

        # Resume the state and the logs
        self.trainer_state = eqx.tree_deserialise_leaves(
            dir_checkpoint / "state.eqx", self.trainer_state
        )
        self.logs = load_json(dir_checkpoint / "logs.json")

        # Verbose
        if verbose:
            print_resumed(
                console=Console(),
                label=label,
                idx_iter=int(self.trainer_state.idx_iter),
                n_iter=self.config_train.n_iter,
                n_logs=len(self.logs),
                dir_checkpoint=dir_checkpoint,
                n_gpu_ckpt=n_gpu_old,
                n_gpu_now=n_gpu_new,
            )
        return self

    @classmethod
    def load(cls, dir_save, env, network_fn, gpu_mesh=None):
        """
        Load a Trainer for new generation
        """
        # Dir management
        dir_save = Path(dir_save)
        network_path = dir_save / "network.eqx"
        if not network_path.exists():
            raise FileNotFoundError(f"No agent to load: {network_path}")

        # Build a fresh trainer from the saved meta/configs
        self = cls._from_disk(dir_save, env, network_fn, gpu_mesh)

        # Restore the neural network into the sharded template
        network = eqx.tree_deserialise_leaves(network_path, self.trainer_state.network)
        self.trainer_state = eqx.tree_at(lambda s: s.network, self.trainer_state, network)
        return self

    # ===== FIT =====

    def fit(self, dir_save=None, checkpoint_every=None, eval_every=None, verbose=True):
        """
        Orchestrate n_iter training iterations
            - init: 
                - save + checkpoint_init
                - one-time XLA warmup (compiles the sharded graphs)
            - loop: 
              - _train_iter 
              - optional eval 
              - log 
              - optional periodic checkpoint
            - end: 
              - save + checkpoint_final
        Returns the accumulated logs.
        """
        ct = self.config_train
        dir_save = Path(dir_save) if dir_save is not None else self.dir_save
        console = Console()
        self._check_trainer_state()
        start = int(self.trainer_state.idx_iter)

        # ===== INIT =====
        dir_save.mkdir(parents=True, exist_ok=True)
        self._save_meta_configs(dir_save)
        if start == 0:
            self.checkpoint(dir_save, label="init")

        # Warmup for pre-compilation before training
        self._warmup(console, verbose=verbose)

        # ===== TRAINING LOOP =====
        progress = make_progress(console, verbose)
        with progress:
            task = progress.add_task("train", total=ct.n_iter, completed=start)
            for idx_iter in range(start, ct.n_iter):
                tic_iter = time.perf_counter()

                # --- Get the current state ---
                s = self.trainer_state

                # --- Key management ---
                key_train, subkey_train = jax.random.split(s.key_train)
                key_eval, subkey_eval = jax.random.split(s.key_eval)

                # --- Training ---
                network, opt_state, log = self._train_iter(
                    network=s.network,
                    opt_state=s.opt_state,
                    key=subkey_train,
                )
                log["idx_iter"] = idx_iter

                # --- Evaluation ---
                eval_metrics = None
                is_first, is_last = idx_iter == start, idx_iter == ct.n_iter - 1
                do_eval = is_first or is_last or (eval_every is not None and idx_iter % eval_every == 0)
                if do_eval:
                    tic_eval = time.perf_counter()
                    with progress.console.status("[cyan]evaluating...", spinner="dots"):
                        metrics = self._evaluate(network=network, key=subkey_eval)
                        jax.block_until_ready(metrics)
                    eval_metrics = {
                        "success": float(metrics["mean_success"]),
                        "score":   float(metrics["mean_score"]),
                        "length":  float(metrics["mean_length"]),
                        "time_eval": time.perf_counter() - tic_eval,
                    }
                log["eval"] = eval_metrics
                log["time_iter"] = time.perf_counter() - tic_iter
                self.logs.append(log)

                # --- Commit new state ---
                self.trainer_state = TrainerState(
                    network=network,
                    opt_state=opt_state,
                    key_train=key_train,
                    key_eval=key_eval,
                    idx_iter=jnp.array(idx_iter + 1, dtype=jnp.int32),
                )

                # --- Checkpoint ---
                if checkpoint_every is not None and (idx_iter + 1) % checkpoint_every == 0:
                    self.checkpoint(dir_save, label=str(idx_iter + 1))

                # --- Per-iter print ---
                if verbose:
                    print_iter(progress.console, idx_iter, log, eval_metrics)

                progress.update(task, advance=1)

        # ===== FINAL =====
        self.checkpoint(dir_save, label="end")
        self.save(dir_save)
        return self.logs

    def _warmup(self, console, verbose=True):
        """
        Compile the sharded XLA graphs once
        """
        s = self.trainer_state
        tic_warm = time.perf_counter()

        # --- Train graph ---
        with status(console, "[bold cyan]Compiling XLA graph for training...", verbose):
            _, key_warm = jax.random.split(s.key_train)
            warm = self._train_iter(network=s.network, opt_state=s.opt_state, key=key_warm)
            jax.block_until_ready(warm)
            del warm

        # --- Eval graph ---
        with status(console, "[bold cyan]Compiling XLA graph for eval...", verbose):
            _, key_warm_eval = jax.random.split(s.key_eval)
            warm_eval = self._evaluate(network=s.network, key=key_warm_eval)
            jax.block_until_ready(warm_eval)
            del warm_eval

        if verbose:
            print_compiled(console, time.perf_counter() - tic_warm)
