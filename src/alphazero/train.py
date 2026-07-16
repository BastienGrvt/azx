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
from .evaluate import evaluate


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
def train_step(network, opt_state, data_minibatch, optimizer, c_value):
    # one gradient step; returns updated network + optimizer state + metrics
    (loss, loss_dict), grads = loss_fn(network, data_minibatch, c_value)
    updates, opt_state = optimizer.update(grads, opt_state, network)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, {"loss": loss, **loss_dict}


# ---- one self-play round, aggregating n_roll calls (jitted via scan) ----
@eqx.filter_jit
def collect(
    env, network, recurrent_fn, key, 
    n_roll, n_batch, n_step, num_sim, 
    discount, temperature, dirichlet_fraction, dirichlet_alpha,
):
    """
    Data collector, does n_roll rollout
    1. Get data from n_roll call of selplay(...)
    2. Merge [n_roll, N, *] -> [N*n_roll, *] with N = B*n_step
    """

    def body_rollout(key, _):
        key, key_sub = jax.random.split(key)
        game = selfplay(
            env=env,
            network=network,
            recurrent_fn=recurrent_fn,
            key=key,
            n_batch=n_batch,
            n_step=n_step,
            num_sim=num_sim, 
            discount=discount, 
            temperature=temperature,
            dirichlet_fraction=dirichlet_fraction,
            dirichlet_alpha=dirichlet_alpha
        )
        return key, game
    
    _, data_roll = jax.lax.scan(body_rollout, key, xs=None, length=n_roll)
    out = {item: data.reshape(-1, *data.shape[2:]) for item, data in data_roll.items()}

    return out

# ---- AlphaZero training loop ----

def train(
    env, network, recurrent_fn, optimizer, opt_state, key,
    n_iter,
    n_roll,
    n_update,
    n_minibatch,
    n_batch,
    n_step,
    num_sim,
    discount=0.9,
    c_value=1.0,
    temperature=1.0,
    dirichlet_fraction=0.25,
    dirichlet_alpha=0.3,
    eval=False,
    eval_every=1,
    eval_n_batch=None,
    eval_n_step=None,
    eval_num_sim=None,
    eval_temperature=0.0,
    eval_dirichlet_fraction=0.0,
    eval_dirichlet_alpha=0.3,
    verbose=True,
):
    """
    AlphaZero training loop
 
    1. Get data from collect(...)
    2. Batch the data in minibatch
    3. Update the neural network with train_step(...)
    4. (optional) Greedy evaluate(...) every `eval_every` iters -> log["eval"]
 
    About the n_*:
        n_iter:      number of data collection
        n_roll:      number of rollout per n_iter
        n_update:    number of passes over the collected data per n_iter
        n_minibatch: number of minibatches per n_update (fixed size, drop_last)
        n_batch:     rollout batch size (parallel games)
        n_step:      number of step per game (rollout horizon)
        num_sim:     MCTS simulations per move
 
    Eval (post-update perf, greedy). Defaults fall back to the train sizes:
        eval:        toggle periodic evaluation
        eval_every:  run eval every k iters (and always on the last iter)
        eval_*:      dedicated (smaller) eval sizes; None -> reuse train value
    """
 
    # ===== INIT =====
    eval_n_batch = eval_n_batch if eval_n_batch is not None else n_batch
    eval_n_step  = eval_n_step  if eval_n_step  is not None else n_step
    eval_num_sim = eval_num_sim if eval_num_sim is not None else num_sim
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
            # Data Collect compilation
            warm = collect(
                env=env,
                network=network,
                recurrent_fn=recurrent_fn,
                key=key_warm,
                n_roll=n_roll,
                n_batch=n_batch,
                n_step=n_step,
                num_sim=num_sim,
                discount=discount, 
                temperature=temperature,
                dirichlet_fraction=dirichlet_fraction,
                dirichlet_alpha=dirichlet_alpha
            )
            jax.block_until_ready(warm)
            # Evaluation compilation
            if eval:
                key, key_warm_eval = jax.random.split(key)
                warm_eval = evaluate(
                    env=env,
                    network=network,
                    recurrent_fn=recurrent_fn,
                    key=key_warm_eval,
                    n_batch=eval_n_batch,
                    n_step=eval_n_step,
                    num_sim=eval_num_sim,
                    temperature=eval_temperature,
                    dirichlet_fraction=eval_dirichlet_fraction,
                    dirichlet_alpha=eval_dirichlet_alpha,
                )
                jax.block_until_ready(warm_eval)
                del warm_eval
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
                env=env,
                network=network,
                recurrent_fn=recurrent_fn,
                key=key_roll,
                n_roll=n_roll,
                n_batch=n_batch,
                n_step=n_step,
                num_sim=num_sim,
                discount=discount, 
                temperature=temperature,
                dirichlet_fraction=dirichlet_fraction,
                dirichlet_alpha=dirichlet_alpha
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
                    network, opt_state, loss = train_step(
                        network=network,
                        opt_state=opt_state,
                        data_minibatch=data_minibatch,
                        optimizer=optimizer,
                        c_value=c_value,
                    )
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
 
            # --- Periodic greedy evaluation (post-update perf) ---
            time_updates = time.perf_counter() - tic_updates
            eval_metrics = None
            if eval and ((idx_iter % eval_every == 0) or (idx_iter == n_iter - 1)):
                key, key_eval = jax.random.split(key)
                tic_eval = time.perf_counter()
                # Spinner below the progress bar while evaluate() computes
                with progress.console.status("[cyan]evaluating...", spinner="dots"):
                    metrics = evaluate(
                        env=env,
                        network=network,
                        recurrent_fn=recurrent_fn,
                        key=key_eval,
                        n_batch=eval_n_batch,
                        n_step=eval_n_step,
                        num_sim=eval_num_sim,
                        temperature=eval_temperature,
                        dirichlet_fraction=eval_dirichlet_fraction,
                        dirichlet_alpha=eval_dirichlet_alpha,
                    )
                    # Let the spinner catch the real time
                    jax.block_until_ready(metrics)
                eval_metrics = {
                    "success": float(metrics["mean_success"]),
                    "score":   float(metrics["mean_score"]),
                    "length":  float(metrics["mean_length"]),
                    "time_eval": time.perf_counter() - tic_eval,
                }
 
            # --- Save iteration metrics ---
            time_iter = time.perf_counter() - tic_iter
            log_roll = log_roll | {
                "time_iter": time_iter,
                "time_updates": time_updates,
                "logs_nn": logs_nn,
                "eval": eval_metrics,
            }
            logs.append(log_roll)
 
            # --- Persistent per-iteration log (ordered: metrics | timing) ---
            if verbose:
                last = logs_nn[-1]
                # line 1: Networks
                progress.console.print(
                    f"[dim]iter {idx_iter:3d}[/dim]  "
                    f"loss=[bold]{last['loss']:.4f}[/bold]  "
                    f"policy={last['policy_loss']:.4f}  "
                    f"value={last['value_loss']:.4f}  "
                    f"val_mean={log_roll['value_mean']:.3f}  "
                    f"[dim]|[/dim]  "
                    f"[cyan]roll {time_roll:4.1f}s[/cyan]  "
                    f"[yellow]upd {time_updates:4.1f}s[/yellow]  "
                    f"[dim]tot {time_iter:4.1f}s[/dim]"
                )
                # line 2: evaluate performance
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

 
    return network, opt_state, logs

