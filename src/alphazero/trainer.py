"""
Trainer: orchestration class wrapping train / save / load.

"""

# System package
from pathlib import Path
import json, dataclasses, time

# GPU package
import jax, jax.numpy as jnp
import equinox as eqx
import optax

# Rich package
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# AlphaZero packages
from .config import Meta, ConfigTrain, ConfigEval, MctsEngine
from .train import train_iter, collect
from .evaluate import evaluate
from .search import make_recurrent_fn



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

    def __init__(self, meta, env, network_fn, config_train, config_eval, dir_save):
        self.meta = meta
        self.env = env
        self.network_fn = network_fn
        self.config_train = config_train
        self.config_eval = config_eval
        self.dir_save = Path(dir_save)
        # Build the reciep
        self.optimizer = _OPT[config_train.optimizer](**config_train.optimizer_kwargs)
        self.recurrent_fn_train = make_recurrent_fn(env, discount=config_train.mcts.discount)
        self.recurrent_fn_eval = make_recurrent_fn(env, discount=config_eval.mcts.discount)
        # Mutable state
        self.trainer_state = None
        self.logs = []

    def init(self):
        # Key management
        # ToDo: manage new jax.random key generation (used old PRNGKey to avoid equinox save conflict)
        key_train = jax.random.PRNGKey(self.config_train.seed)
        key_eval = jax.random.PRNGKey(self.config_eval.seed)
        # Setup fresh network and optimizer
        _, key_net = jax.random.split(key_train)
        network = self.network_fn(self.env, key_net)
        opt_state = self.optimizer.init(eqx.filter(network, eqx.is_array))
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
    def resume(cls, dir_save, env, network_fn, label="end", verbose=True):
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

        # Build a fesh trainer
        config_train = ConfigTrain(**config["train"])
        config_eval = ConfigEval(**config["eval"])
        self = cls(meta, env, network_fn, config_train, config_eval, dir_save)
        self.init()

        # Resume the state and the logs
        self.trainer_state = eqx.tree_deserialise_leaves(dir_checkpoint / "state.eqx", self.trainer_state)
        with open(dir_checkpoint / "logs.json") as f:
            self.logs = json.load(f)

        # Verbose
        if verbose:
            idx = int(self.trainer_state.idx_iter)
            Console().print(
                f"[green]✓ Resumed[/green] from [bold]checkpoint_{label}[/bold]  "
                f"[dim]|[/dim]  iter=[bold]{idx}[/bold]/{config_train.n_iter}  "
                f"[dim]|[/dim]  {len(self.logs)} logs restored  "
                f"[dim]|[/dim]  {dir_checkpoint}"
            )
        return self


    @classmethod
    def load(cls, dir_save, env, network_fn):
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

        # Build a fesh trainer
        config_train = ConfigTrain(**config["train"])
        config_eval = ConfigEval(**config["eval"])
        self = cls(meta, env, network_fn, config_train, config_eval, dir_save)
        self.init()

        # Restore the neural network
        network = eqx.tree_deserialise_leaves(network_path, self.trainer_state.network)
        self.trainer_state = eqx.tree_replace(self.trainer_state, network=network)
        return self


    def fit(self, dir_save=None, checkpoint=True, checkpoint_every=1, eval=False, eval_every=1, verbose=True):
            """
            Orchestrate n_iter training iterations (mutates self.trainer_state).
              - init disk (meta/configs) + checkpoint_init
              - one-time XLA warmup
              - loop: train_iter -> optional greedy eval -> log -> periodic checkpoint
              - end: checkpoint_end + save root artifact (network/logs)
            Returns the accumulated logs.
            """
            ct, ce = self.config_train, self.config_eval
            dir_save = Path(dir_save) if dir_save is not None else self.dir_save
            console = Console()
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

            # ===== WARMUP =====
            if verbose:
                with console.status("[bold cyan]Compiling XLA graph...", spinner="dots"):
                    tic_warm = time.perf_counter()
                    key_train, key_warm = jax.random.split(s.key_train)
                    warm = collect(
                        env=self.env, network=s.network, recurrent_fn=self.recurrent_fn_train,
                        key=key_warm,
                        n_roll=ct.n_roll, n_batch=ct.n_batch, n_step=ct.n_step, num_sim=ct.mcts.num_sim,
                        discount=ct.mcts.discount,
                        temperature=ct.mcts.temperature,
                        dirichlet_fraction=ct.mcts.dirichlet_fraction,
                        dirichlet_alpha=ct.mcts.dirichlet_alpha,
                    )
                    jax.block_until_ready(warm)
                    del warm
                    if eval:
                        key_eval, key_warm_eval = jax.random.split(s.key_eval)
                        warm_eval = evaluate(
                            env=self.env, network=s.network, recurrent_fn=self.recurrent_fn_eval,
                            key=key_warm_eval,
                            n_batch=ce.n_batch, n_step=ce.n_step, num_sim=ce.mcts.num_sim,
                            discount=ce.mcts.discount,
                            temperature=ce.mcts.temperature,
                            dirichlet_fraction=ce.mcts.dirichlet_fraction,
                            dirichlet_alpha=ce.mcts.dirichlet_alpha,
                        )
                        jax.block_until_ready(warm_eval)
                        del warm_eval
                console.print(f"[green]✓ Compiled in {time.perf_counter() - tic_warm:.1f}s")

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
                    log = None
                    network, opt_state, log = train_iter(
                        # Trainer params
                        env=self.env,
                        recurrent_fn=self.recurrent_fn_train,
                        optimizer=self.optimizer,
                        # Trainer state params
                        network=s.network,
                        opt_state=s.opt_state,
                        key=subkey_train,
                        # Config params
                        n_roll=ct.n_roll,
                        n_update=ct.n_update,
                        n_minibatch=ct.n_minibatch,
                        n_batch=ct.n_batch,
                        n_step=ct.n_step,
                        num_sim=ct.mcts.num_sim,
                        discount=ct.mcts.discount,
                        c_value=ct.c_value,
                        temperature=ct.mcts.temperature,
                        dirichlet_fraction=ct.mcts.dirichlet_fraction,
                        dirichlet_alpha=ct.mcts.dirichlet_alpha,
                    )
                    log["idx_iter"] = idx_iter

                    # --- Evaluation ---
                    eval_metrics = None
                    if eval and ((idx_iter % eval_every == 0) or (idx_iter == ct.n_iter - 1)):
                        tic_eval = time.perf_counter()
                        with progress.console.status("[cyan]evaluating...", spinner="dots"):
                            metrics = evaluate(
                                # Trainer params
                                env=self.env,
                                recurrent_fn=self.recurrent_fn_eval,
                                network=network,
                                key=subkey_eval,
                                # Config params
                                n_batch=ce.n_batch,
                                n_step=ce.n_step,
                                num_sim=ce.mcts.num_sim,
                                discount=ce.mcts.discount,
                                temperature=ce.mcts.temperature,
                                dirichlet_fraction=ce.mcts.dirichlet_fraction,
                                dirichlet_alpha=ce.mcts.dirichlet_alpha,
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
                    if checkpoint and ((idx_iter + 1) % checkpoint_every == 0):
                        self.checkpoint(dir_save, label=str(idx_iter + 1))

                    # --- Per-iter print ---
                    if verbose:
                        last = log["logs_nn"][-1]
                        progress.console.print(
                            f"[dim]iter {idx_iter:3d}[/dim]  "
                            f"loss=[bold]{last['loss']:.4f}[/bold]  "
                            f"policy={last['policy_loss']:.4f}  "
                            f"value={last['value_loss']:.4f}  "
                            f"val_mean={log['value_mean']:.3f}  "
                            f"[dim]|[/dim]  "
                            f"[cyan]roll {log['time_roll']:4.1f}s[/cyan]  "
                            f"[yellow]upd {log['time_updates']:4.1f}s[/yellow]  "
                            f"[dim]tot {log['time_iter']:4.1f}s[/dim]"
                        )
                        if eval_metrics is not None:
                            progress.console.print(
                                f"          [green]└─ eval[/green]  "
                                f"succ=[bold green]{eval_metrics['success']:.2f}[/bold green]  "
                                f"score={eval_metrics['score']:.3f}  "
                                f"len={eval_metrics['length']:.1f}  "
                                f"[dim]|[/dim]  "
                                f"[magenta]eval {eval_metrics['time_eval']:4.1f}s[/magenta]"
                            )
                    progress.update(task, advance=1)

            # ===== FINAL =====
            self.checkpoint(dir_save, label="end")
            self.save(dir_save)
            return self.logs



