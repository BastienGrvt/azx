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
import equinox as eqx
import optax
import time

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
def collect(env, network, recurrent_fn, key, B, n_roll, num_sim, n_step, discount=0.99):
    """
    Data collector, does n_roll rollout
    1. Get data from n_roll call of selplay(...)
    2. Merge [n_roll, N, *] -> [N*n_roll, *] with N = B*n_step
    """

    def body_rollout(key, _):
        key, key_sub = jax.random.split(key)
        game = selfplay(env, network, recurrent_fn, key_sub, B, num_sim, n_step, discount)  # [N, *]
        return key, game
    
    _, data_roll = jax.lax.scan(body_rollout, key, xs=None, length=n_roll)
    out = {item: data.reshape(-1, *data.shape[2:]) for item, data in data_roll.items()}

    return out

# ---- Main AlphaZero loop ----
# TODO: logs and time
def train(
    env, network, recurrent_fn, optimizer, opt_state, key,
    n_iter, n_roll, n_update, n_minibatch,
    n_batch, n_step, num_sim,
    c_value=1.0, discount=0.99, verbose=True,
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
    logs = []
    for idx_iter in range(n_iter):
        tic_iter = time.perf_counter()
        key, key_roll = jax.random.split(key)
        # Get data
        if verbose:
            print(f"=== Starting iteration: {idx_iter} ===")
            print(f"Playing virtual games...")
        tic_roll = time.perf_counter()
        data_roll = collect(
            env, network=network, recurrent_fn=recurrent_fn, key=key_roll,
            B=n_batch, n_roll=n_roll, num_sim=num_sim, n_step=n_step, discount=discount,
        )
        time_roll = tic_roll - time.perf_counter()
        if verbose:
            print(f"Virtal games ended in {time_roll:.1f}s")
        # N = n_step*n_batch*n_roll
        N = data_roll["value"].shape[0]
        size_minibatch = N // n_minibatch
        for idx_update in range(n_update):
            tic_update = time.perf_counter()
            key, key_perm = jax.random.split(key)
            idx_perm = jax.random.permutation(key_perm, N)
            idx_perm_cut = idx_perm[:n_minibatch * size_minibatch]
            for idx_mb in idx_perm_cut.reshape(n_minibatch, size_minibatch):
                data_minibatch = {k: v[idx_mb] for k, v in data_roll.items()}
                network, opt_state, loss = train_step(network, opt_state, data_minibatch, optimizer, c_value)
                logs.append({
                    "iter": idx_iter,
                    "minibatch": idx_mb
                    "update": idx_update,
                    "loss": loss
                    })
    return network, opt_state



