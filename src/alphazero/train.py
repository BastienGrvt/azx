"""
ToDo: new comment header version

AlphaZero training step (single gradient pass over a self-play batch).

Targets come from selfplay():
  observation [N,obs]  policy [N,A]  value [N]
selfplay() already flattens time/batch (continuous rollout, no padding), so
every sample is a valid transition -> no masking needed here.

Loss = policy_loss + c_value * value_loss
  policy_loss : cross-entropy(network logits, MCTS visit distribution)
  value_loss  : MSE(network value, discounted return)
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import time

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


from .selfplay import selfplay


@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(network, data_minibatch, c_value):
    # batch already flat [mb_size, *] -> predict directly
    obs = data_minibatch["observation"]
    z_policy = data_minibatch["policy"]
    z_value = data_minibatch["value"]

    # --- Network prediction ---
    y_policy, y_value = eqx.filter_vmap(network)(obs)

    # --- Loss ---
    # Policy: cross-entropy
    policy_loss = optax.softmax_cross_entropy(y_policy, z_policy).mean()
    # Value: mean squared
    value_loss = ((y_value - z_value) ** 2).mean()

    loss = policy_loss + c_value * value_loss
    return loss, {"policy_loss": policy_loss, "value_loss": value_loss}


@eqx.filter_jit
def train_step(network, opt_state, data_minibatch, optimizer, c_value=1.0):
    # one gradient step; returns updated network + optimizer state + metrics
    (loss, loss_dict), grads = loss_fn(network, data_minibatch, c_value)
    updates, opt_state = optimizer.update(grads, opt_state, network)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, {"loss": loss, **loss_dict}


# ---- one self-play round, aggregating n_roll calls (jitted via scan) ----
@eqx.filter_jit
def collect(env, network, recurrent_fn, key, n_roll, n_batch, n_step, num_sim, discount):
    """
    Data collector, does n_roll rollout
    1. Get data from n_roll call of selplay(...)
    2. Merge [n_roll, N, *] -> [N*n_roll, *] with N = B*n_step
    """

    def body_rollout(key, _):
        key, key_sub = jax.random.split(key)
        game = selfplay(env, network, recurrent_fn, key_sub, n_batch, n_step, num_sim, discount)
        return key, game
    
    _, data_roll = jax.lax.scan(body_rollout, key, xs=None, length=n_roll)
    out = {item: data.reshape(-1, *data.shape[2:]) for item, data in data_roll.items()}

    return out

# ---- AlphaZero training loop ----

# def train(
#     env, network, recurrent_fn, optimizer, opt_state, key,
#     n_iter, n_roll, n_update, n_minibatch,
#     n_batch, n_step, num_sim,
#     discount, c_value=1.0, verbose=True,
# ):
#     """
#     AlphaZero training loop
#
#     1. Get data from collect(...)
#     2. Batch the data in minibatch
#     3. Update the neural network with train_step(...)
#
#     About the n_*:
#         n_iter:      number of data collection
#         n_roll:      number of rollout per n_iter
#         n_update:    number of passes over the collected data per n_iter
#         n_minibatch: number of minibatches per n_update (fixed size, drop_last)
#         n_batch:     rollout batch size (parallel games)
#         n_step:      number of step per game (rollout horizon)
#         num_sim:     MCTS simulations per move
#     """
#     logs = []
#     key, _ = jax.random.split(key)
#     for idx_iter in range(n_iter):
#
#         # --- Initialization ---
#         key, key_roll = jax.random.split(key)
#         log_roll = {"idx_iter": idx_iter}
#         if verbose:
#             print(f"=== Starting iteration: {idx_iter} ===")
#         tic_iter = time.perf_counter()
#
#         # --- Data collection (self-play) ---
#         if verbose:
#             print("Playing virtual games...")
#         tic_roll = time.perf_counter()
#         # Self-play
#         data_roll = collect(
#             env=env, network=network, recurrent_fn=recurrent_fn, key=key_roll,
#             n_roll=n_roll, n_batch=n_batch, n_step=n_step, num_sim=num_sim,
#             discount=discount,
#         )
#         # Logs
#         time_roll = time.perf_counter() - tic_roll
#         log_roll = log_roll | {
#             "time_roll": time_roll,
#             "value_mean": float(data_roll["value"].mean()),
#         }
#         if verbose:
#             print(f"Virtual games ended in {time_roll:.1f}s")
#             print("Starting neural network update...")
#
#         # --- Mini-batch setup ---
#         # N = n_step*n_batch*n_roll
#         N = data_roll["value"].shape[0]
#         size_minibatch = N // n_minibatch
#
#         # --- Neural Network updates ---
#         tic_updates = time.perf_counter()
#         logs_nn = []
#         for idx_update in range(n_update):
#             key, key_perm = jax.random.split(key)
#             # Permutation
#             idx_perm = jax.random.permutation(key_perm, N)
#             idx_perm_cut = idx_perm[:n_minibatch * size_minibatch]
#             tic_update = time.perf_counter()
#             # Update over minibatchs
#             mb_metrics = [] 
#             for idx_mb in idx_perm_cut.reshape(n_minibatch, size_minibatch):
#                 data_minibatch = {k: v[idx_mb] for k, v in data_roll.items()}
#                 network, opt_state, loss = train_step(network, opt_state, data_minibatch, optimizer, c_value)
#                 mb_metrics.append(jax.device_get(loss))
#             # Logs
#             time_update = time.perf_counter() - tic_update
#             mean_loss = {k: float(np.mean([m[k] for m in mb_metrics])) for k in mb_metrics[0]}
#             logs_nn.append({
#                 "idx_iter": idx_iter,
#                 "idx_update": idx_update,
#                 "time_update": time_update,
#                 **mean_loss,
#             })
#         # Logs
#         time_updates = time.perf_counter() - tic_updates
#         time_iter = time.perf_counter() - tic_iter
#         log_roll = log_roll | {
#             "time_iter": time_iter,
#             "time_updates": time_updates,
#             "logs_nn": logs_nn,
#         }
#         logs.append(log_roll)
#         if verbose:
#             print(f"Neural Network updated in {time_updates:.1f}s")
#             print(f"Iteration ended in {time_iter:.1f}s\n")
#
#     return network, opt_state, logs


def train(
    env, network, recurrent_fn, optimizer, opt_state, key,
    n_iter, n_roll, n_update, n_minibatch,
    n_batch, n_step, num_sim,
    discount, c_value=1.0, verbose=True,
):
    """
    AlphaZero training loop

    1. Get data from collect(...)
    2. Batch the data in minibatch
    3. Update the neural network with train_step(...)

    About the n_*:
        n_iter:      number of data collection
        n_roll:      number of rollout per n_iter
        n_update:    number of passes over the collected data per n_iter
        n_minibatch: number of minibatches per n_update (fixed size, drop_last)
        n_batch:     rollout batch size (parallel games)
        n_step:      number of step per game (rollout horizon)
        num_sim:     MCTS simulations per move
    """

    # ===== INIT =====
    logs = []
    console = Console()
    key, _ = jax.random.split(key)

    # ===== WARMUP =====
    # ~Trigger compilation XLA
    # ~Show compilation time
    if verbose:
        with console.status("[bold cyan]Compiling XLA graph...", spinner="dots"):
            tic_warm = time.perf_counter()
            key, key_warm = jax.random.split(key)
            warm = collect(
                env=env, network=network, recurrent_fn=recurrent_fn, key=key_warm,
                n_roll=n_roll, n_batch=n_batch, n_step=n_step, num_sim=num_sim,
                discount=discount,
            )
            jax.block_until_ready(warm)  # force compile + first run to actually finish
        console.print(f"[green]✓ Compiled in {time.perf_counter() - tic_warm:.1f}s")
        del warm

    # ===== TRAINING LOOP =====

    # --- Progress Bar ---
    progress = Progress(
        TextColumn("[bold blue]iter {task.completed}/{task.total}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not verbose,
    )

    # --- Main loop ---
    with progress:
        task = progress.add_task("train", total=n_iter)
        for idx_iter in range(n_iter):

            # --- Initialization ---
            key, key_roll = jax.random.split(key)
            log_roll = {"idx_iter": idx_iter}
            tic_iter = time.perf_counter()

            # --- Data collection (self-play) ---
            tic_roll = time.perf_counter()
            data_roll = collect(
                env=env, network=network, recurrent_fn=recurrent_fn, key=key_roll,
                n_roll=n_roll, n_batch=n_batch, n_step=n_step, num_sim=num_sim,
                discount=discount,
            )
            time_roll = time.perf_counter() - tic_roll
            log_roll = log_roll | {
                "time_roll": time_roll,
                "value_mean": float(data_roll["value"].mean()),
            }

            # --- Mini-batch setup ---
            N = data_roll["value"].shape[0]
            size_minibatch = N // n_minibatch

            # --- Neural Network updates ---
            tic_updates = time.perf_counter()
            logs_nn = []
            for idx_update in range(n_update):
                key, key_perm = jax.random.split(key)
                idx_perm = jax.random.permutation(key_perm, N)
                idx_perm_cut = idx_perm[:n_minibatch * size_minibatch]
                tic_update = time.perf_counter()

                mb_metrics = []
                for idx_mb in idx_perm_cut.reshape(n_minibatch, size_minibatch):
                    data_minibatch = {k: v[idx_mb] for k, v in data_roll.items()}
                    network, opt_state, loss = train_step(network, opt_state, data_minibatch, optimizer, c_value)
                    mb_metrics.append(jax.device_get(loss))

                time_update = time.perf_counter() - tic_update
                mean_loss = {k: float(np.mean([m[k] for m in mb_metrics])) for k in mb_metrics[0]}
                logs_nn.append({
                    "idx_iter": idx_iter,
                    "idx_update": idx_update,
                    "time_update": time_update,
                    **mean_loss,
                })

            time_updates = time.perf_counter() - tic_updates
            time_iter = time.perf_counter() - tic_iter
            log_roll = log_roll | {
                "time_iter": time_iter,
                "time_updates": time_updates,
                "logs_nn": logs_nn,
            }
            logs.append(log_roll)

            # --- Persistent per-iteration log line (stays on screen) ---
            if verbose:
                last = logs_nn[-1]
                progress.console.print(
                    f"[dim]iter {idx_iter:3d}[/dim]  "
                    f"loss=[bold]{last['loss']:.4f}[/bold]  "
                    f"policy={last['policy_loss']:.4f}  "
                    f"value={last['value_loss']:.4f}  "
                    f"val_mean={log_roll['value_mean']:.3f}  "
                    f"[cyan]roll={time_roll:.1f}s[/cyan]  "
                    f"[yellow]upd={time_updates:.1f}s[/yellow]  "
                    f"[dim](tot {time_iter:.1f}s)[/dim]"
                )
            progress.update(task, advance=1)

    return network, opt_state, logs
