"""
Trainer: orchestration class wrapping train / save / load.

"""

# System package
from pathlib import Path
from functools import partial
import json, dataclasses, time

# GPU package
import jax, jax.numpy as jnp
import equinox as eqx
import optax

# Rich package
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# AlphaZero packages
from .config import Meta, ConfigTrain, ConfigEval
from .train import train_iter, make_collect_global, make_update_global
from .evaluate import evaluate, make_evaluate_global
from .search import make_recurrent_fn, make_search_fn


_OPT = {"adam": optax.adam, "sgd": optax.sgd, "adamw": optax.adamw}


# Logs already jsonable (no JAX objects)

# def _to_jsonable(x):
#     if isinstance(x, (np.floating, np.integer)): return x.item()
#     if hasattr(x, "item"): return x.item()
#     if hasattr(x, "tolist"): return x.tolist()
#     raise TypeError(f"not JSON serializable: {type(x)}")

def _to_jsonable(x):
    return x



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
            # MCTS mechanics
            self.recurrent_fn_train, self.search_fn_train = self._make_mcts(config_train)
            self.recurrent_fn_eval, self.search_fn_eval = self._make_mcts(config_eval)
            # Device mesh and sharded builders
            self._make_gpu(gpu_mesh)
            # Mutable state
            self.trainer_state = None
            self.logs = []



    def _make_mcts(self, config):
        """Build the MCTS pair (dynamics + policy) for a given config."""
        recurrent_fn = make_recurrent_fn(self.env, discount=config.mcts.discount)
        search_fn = make_search_fn(
            kind=config.mcts.kind,
            num_sim=config.mcts.num_sim,
            **config.mcts.policy_kwargs,
        )
        return recurrent_fn, search_fn


    def _make_gpu(self, gpu_mesh=None):
        """
        Set up the device mesh and build the three sharded stages
        Config sizes are GLOBAL, per-device sizes are derived
        """
        ct, ce = self.config_train, self.config_eval

        # --- Mesh ---
        if gpu_mesh is not None:
            self.gpu_mesh = gpu_mesh
        else:
            self.gpu_mesh = jax.make_mesh(
                (jax.device_count(),), ("batch",),
                axis_types=(jax.sharding.AxisType.Auto,),
            )
        jax.set_mesh(self.gpu_mesh)
        self.gpu_axis = self.gpu_mesh.axis_names[0]
        self.n_gpu = self.gpu_mesh.shape[self.gpu_axis]

        # --- Host Trainer parameters ---
        self._check_shapes()
        self.n_batch_local_train = ct.n_batch // self.n_gpu
        self.n_batch_local_eval = ce.n_batch // self.n_gpu
        self.n_minibatch_local = ct.n_minibatch // self.n_gpu
        self.N_local = ct.n_roll * ct.n_step * self.n_batch_local_train
        self.size_minibatch_local = self.N_local // self.n_minibatch_local

        # --- Sharded builders ---
        self.collect_global = make_collect_global(
            gpu_mesh=self.gpu_mesh, gpu_axis=self.gpu_axis,
            env=self.env,
            recurrent_fn=self.recurrent_fn_train,
            search_fn=self.search_fn_train,
            n_batch_local=self.n_batch_local_train,
            n_roll=ct.n_roll,
            n_step=ct.n_step,
            discount=ct.mcts.discount,
        )
        self.update_global = make_update_global(
            gpu_mesh=self.gpu_mesh,
            gpu_axis=self.gpu_axis,
            optimizer=self.optimizer,
            n_update=ct.n_update,
            n_minibatch_local=self.n_minibatch_local,
            size_minibatch_local=self.size_minibatch_local,
            c_value=ct.c_value,
        )
        self.evaluate_global = make_evaluate_global(
            gpu_mesh=self.gpu_mesh,
            gpu_axis=self.gpu_axis,
            env=self.env,
            recurrent_fn=self.recurrent_fn_eval,
            search_fn=self.search_fn_eval,
            n_batch_local=self.n_batch_local_eval,
            n_step=ce.n_step,
            discount_score=ce.discount_score,
        )
        # --- Host side ---
        self.train_iter_fn = partial(
            train_iter,
            gpu_mesh=self.gpu_mesh,
            gpu_axis=self.gpu_axis,
            collect_global=self.collect_global,
            update_global=self.update_global,
        )
        self.eval_fn = partial(
            evaluate,
            gpu_mesh=self.gpu_mesh,
            gpu_axis=self.gpu_axis,
            evaluate_global=self.evaluate_global,
        )

    def _check_shapes(self):
            """
            Validate the `n_*` compabilities:
                - Hard errors: sharding requires exact divisibility (shapes must be static)
                - Warning: leftover samples silently dropped from every update (mini batch size)
            """
            n_gpu = self.n_gpu
            ct, ce = self.config_train, self.config_eval
            problems = []

            # --- Hard: batch and minibatch counts must split across devices ---
            if ct.n_batch % n_gpu:
                problems.append(f"train n_batch={ct.n_batch} % n_gpu={n_gpu}")
            if ce.n_batch % n_gpu:
                problems.append(f"eval n_batch={ce.n_batch} % n_gpu={n_gpu}")
            if ct.n_minibatch % n_gpu:
                problems.append(f"n_minibatch={ct.n_minibatch} % n_gpu={n_gpu}")
            if problems:
                raise ValueError("sharding divisibility violated: " + "; ".join(problems))

            # --- Soft: leftover samples dropped by the minibatch split ---
            N_local = ct.n_roll * ct.n_step * (ct.n_batch // n_gpu)
            n_minibatch_local = ct.n_minibatch // n_gpu
            rem = N_local % n_minibatch_local
            if rem:
                size_minibatch = N_local // n_minibatch_local
                Console().print(
                    f"[yellow]⚠ n_minibatch_local={n_minibatch_local} does not divide "
                    f"N_local={N_local} (n_roll·n_step·n_batch/n_gpu).[/yellow]  "
                    f"size_minibatch={size_minibatch}, "
                    f"[bold]{rem}[/bold] samples dropped per update per device "
                    f"([dim]{100 * rem / N_local:.2f}% of data[/dim])."
                )


    def _check_trainer_state(self):
            """
            Check trainer_state integrity before `fit(self)`
            """
            s = self.trainer_state
            if s is None:
                raise RuntimeError(
                    "trainer_state is `None`, please call .init() for a fresh run, "
                    "or Trainer.resume(...) to continue from a checkpoint"
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
            # Replicate across devices (P() = same copy everywhere)
            network, opt_state = eqx.filter_shard(
                (network, opt_state),
                jax.NamedSharding(self.gpu_mesh, jax.P()),
            )
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


    def save(self, dir_save=None):
        # Dir management
        dir_save = Path(dir_save) if dir_save is not None else self.dir_save
        dir_save.mkdir(parents=True, exist_ok=True)
        # Save metadata
        with open(dir_save / "meta.json", "w") as f:
            json.dump(dataclasses.asdict(self.meta), f, indent=2)
        # Save configs
        with open(dir_save / "configs.json", "w") as f:
            json.dump({
                "train": dataclasses.asdict(self.config_train),
                "eval": dataclasses.asdict(self.config_eval),
            }, f, indent=2)
        # Save the logs
        with open(dir_save / "logs.json", "w") as f:
            json.dump(self.logs, f, indent=2, default=_to_jsonable)
        # Save the network
        eqx.tree_serialise_leaves(dir_save / "network.eqx", self.trainer_state.network)

    def checkpoint(self, dir_save=None, label=None):
        # Dir management
        dir_save = Path(dir_save) if dir_save is not None else self.dir_save
        name = label if label is not None else str(int(self.trainer_state.idx_iter))
        dir_checkpoint = dir_save / "checkpoints" / f"checkpoint_{name}"
        dir_checkpoint.mkdir(parents=True, exist_ok=True)
        # Save the checkpoint (state and logs)
        eqx.tree_serialise_leaves(dir_checkpoint / "state.eqx", self.trainer_state)
        with open(dir_checkpoint / "logs.json", "w") as f:
            json.dump(self.logs, f, indent=2, default=_to_jsonable)

    @classmethod
    def resume(cls, dir_save, env, network_fn, gpu_mesh=None, label="end", verbose=True):
        # Dir management
        dir_save = Path(dir_save)
        dir_checkpoint = dir_save / "checkpoints" / f"checkpoint_{label}"
        if not dir_checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {dir_checkpoint}")

        # Get Metadata and configs
        with open(dir_save / "meta.json") as f:
            meta = Meta(**json.load(f))
        with open(dir_save / "configs.json") as f:
            config = json.load(f)

        # Build a fresh trainer
        config_train = ConfigTrain(**config["train"])
        config_eval = ConfigEval(**config["eval"])
        self = cls(
            meta=meta, 
            env=env, 
            network_fn=network_fn, 
            config_train=config_train, 
            config_eval=config_eval, 
            dir_save=dir_save,
            gpu_mesh=gpu_mesh
        )

        # Init the trainer
        self.init()

        # Resume the state and the logs
        self.trainer_state = eqx.tree_deserialise_leaves(dir_checkpoint / "state.eqx", self.trainer_state)
        with open(dir_checkpoint / "logs.json") as f:
            self.logs = json.load(f)

        # Verbose
        if verbose:
            console = Console()
            idx = int(self.trainer_state.idx_iter)
            if meta.n_gpu != self.n_gpu:
                console.print(f"[yellow]⚠ checkpoint trained on {meta.n_gpu} device(s), resuming on {self.n_gpu} "
                              f"-- local minibatch sizes differ, training trajectory will not match[/yellow]")
            console.print(
                f"[green]✓ Resumed[/green] from [bold]checkpoint_{label}[/bold]  "
                f"[dim]|[/dim]  iter=[bold]{idx}[/bold]/{config_train.n_iter}  "
                f"[dim]|[/dim]  {len(self.logs)} logs restored  "
                f"[dim]|[/dim]  {dir_checkpoint}"
            )
        return self


    @classmethod
    def load(cls, dir_save, env, network_fn, gpu_mesh=None):
        # Dir management
        dir_save = Path(dir_save)
        network_path = dir_save / "network.eqx"
        if not network_path.exists():
            raise FileNotFoundError(f"No agent to load: {network_path}")

        # Get Metadata and configs
        with open(dir_save / "configs.json") as f:
            config = json.load(f)
        with open(dir_save / "meta.json") as f:
            meta = Meta(**json.load(f))

        # Build a fresh trainer
        config_train = ConfigTrain(**config["train"])
        config_eval = ConfigEval(**config["eval"])
        self = cls(
            meta=meta,
            env=env,
            network_fn=network_fn,
            config_train=config_train,
            config_eval=config_eval,
            dir_save=dir_save,
            gpu_mesh=gpu_mesh,
        )

        # Init the trainer (builds the state template with the right sharding)
        self.init()

        # Restore the neural network into the sharded template
        network = eqx.tree_deserialise_leaves(network_path, self.trainer_state.network)
        self.trainer_state = eqx.tree_at(lambda s: s.network, self.trainer_state, network)
        return self



    def fit(self, dir_save=None, checkpoint_every=None, eval_every=None, verbose=True):
            """
            Orchestrate n_iter training iterations (mutates self.trainer_state).
              - init disk (meta/configs) + checkpoint_init
              - one-time XLA warmup (compiles the sharded graphs)
              - loop: train_iter -> optional greedy eval -> log -> periodic checkpoint
              - end: checkpoint_end + save root artifact (network/logs)
            Returns the accumulated logs.
            """
            ct, ce = self.config_train, self.config_eval
            dir_save = Path(dir_save) if dir_save is not None else self.dir_save
            console = Console()
            self._check_trainer_state()
            s = self.trainer_state
            start = int(s.idx_iter)

            # ===== INIT SAVE =====
            dir_save.mkdir(parents=True, exist_ok=True)
            with open(dir_save / "meta.json", "w") as f:
                json.dump(dataclasses.asdict(self.meta), f, indent=2)
            with open(dir_save / "configs.json", "w") as f:
                json.dump({"train": dataclasses.asdict(ct), "eval": dataclasses.asdict(ce)}, f, indent=2)
            if start == 0:
                self.checkpoint(dir_save, label="init")

            # Warmup for pre-compilation before training
            self._warmup(s.key_train, s.key_eval, console, verbose=verbose)

            # ===== TRAINING LOOP =====
            progress = Progress(
                TextColumn("[bold blue]iter {task.completed}/{task.total}"),
                BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
                console=console, disable=not verbose,
            )
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
                    network, opt_state, log = self.train_iter_fn(
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
                            metrics = self.eval_fn(
                                network=network,
                                key=subkey_eval,
                            )
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
                        log_nn_mean = log["logs_nn_mean"]
                        has_eval = eval_metrics is not None
                        branch = "├─" if has_eval else "└─"

                        progress.console.print(
                            f"[bold blue]iter {idx_iter:3d}[/bold blue]  [dim]tot {log['time_iter']:4.1f}s[/dim]"
                        )
                        progress.console.print(
                            f"[dim]{branch}[/dim] [green]train[/green]  "
                            f"loss=[bold]{log_nn_mean['loss']:.4f}[/bold]  "
                            f"policy={log_nn_mean['policy_loss']:.4f}  "
                            f"value={log_nn_mean['value_loss']:.4f}  "
                            f"val_mean={log['value_mean']:.3f}  "
                            f"[dim]|[/dim]  "
                            f"[cyan]roll {log['time_roll']:4.1f}s[/cyan]  "
                            f"[yellow]upd {log['time_updates']:4.1f}s[/yellow]"
                        )
                        if has_eval:
                            progress.console.print(
                                f"[dim]└─[/dim] [green]eval [/green]  "
                                f"succ=[bold green]{eval_metrics['success']:.2f}[/bold green]  "
                                f"score={eval_metrics['score']:.3f}  "
                                f"len={eval_metrics['length']:.1f}  "
                                f"[dim]|[/dim]  "
                                f"[magenta]eval {eval_metrics['time_eval']:4.1f}s[/magenta]"
                            )
                        progress.console.print()

                    progress.update(task, advance=1)

            # ===== FINAL =====
            self.checkpoint(dir_save, label="end")
            self.save(dir_save)
            return self.logs


    def _warmup(self, key_train, key_eval, console, verbose=True):
            """
            Compile the sharded XLA graphs once
            """
            s = self.trainer_state
            tic_warm = time.perf_counter()

            def _status(msg):
                return console.status(msg, spinner="dots") if verbose else nullcontext()

            # --- Train graph ---
            with _status("[bold cyan]Compiling XLA graph for training..."):
                _, key_warm = jax.random.split(key_train)
                warm = self.train_iter_fn(
                    key=key_warm,
                    network=s.network,
                    opt_state=s.opt_state,
                )
                jax.block_until_ready(warm)
                del warm

            # --- Eval graph ---
            with _status("[bold cyan]Compiling XLA graph for eval..."):
                _, key_warm_eval = jax.random.split(key_eval)
                warm_eval = self.eval_fn(
                    key=key_warm_eval,
                    network=s.network,
                )
                jax.block_until_ready(warm_eval)
                del warm_eval

            if verbose:
                console.print(f"[green]✓ Compiled in {time.perf_counter() - tic_warm:.1f}s")




