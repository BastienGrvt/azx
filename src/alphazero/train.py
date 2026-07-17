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
import numpy as np
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
def update_step(network, opt_state, data_minibatch, optimizer, c_value):
    # one gradient step; returns updated network + optimizer state + metrics
    (loss, loss_dict), grads = loss_fn(network, data_minibatch, c_value)
    updates, opt_state = optimizer.update(grads, opt_state, network)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, {"loss": loss, **loss_dict}


# @eqx.filter_jit
# def run_updates(network, opt_state, data_roll, optimizer, key_update,
#                 n_update, n_minibatch, size_minibatch, c_value):
#
#     def body_update(carry, idx_update):
#         network, opt_state = carry
#         subkey = jax.random.fold_in(key_update, idx_update)
#         idx_perm = jax.random.permutation(subkey, n_minibatch * size_minibatch)
#         idx_mb_all = idx_perm.reshape(n_minibatch, size_minibatch)
#
#         def body_minibatch(carry, idx_mb):
#             network, opt_state = carry
#             data_mb = {k: v[idx_mb] for k, v in data_roll.items()}
#             network, opt_state, loss = update_step(
#                 network, opt_state, data_mb, optimizer, c_value)
#             return (network, opt_state), loss
#
#         (network, opt_state), losses = jax.lax.scan(
#             body_minibatch, (network, opt_state), idx_mb_all)
#         # mean over minibatches -> per-update loss dict
#         mean_loss = {k: v.mean() for k, v in losses.items()}
#         return (network, opt_state), mean_loss
#
#     (network, opt_state), logs_nn = jax.lax.scan(
#         body_update, (network, opt_state), jnp.arange(n_update))
#     return network, opt_state, logs_nn   # logs_nn: dict of [n_update] arrays



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
            key=key_sub,
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

def train_iter(
    env, network, recurrent_fn, optimizer, opt_state, key,
    n_roll, n_update, n_minibatch, n_batch, n_step, num_sim,
    discount, c_value,
    temperature, dirichlet_fraction, dirichlet_alpha,
):
    """
    One training iteration (no verbose, no eval, no warmup):
      1. collect self-play data (n_roll rollouts)
      2. n_update passes over the data, each split into n_minibatch minibatches
    Returns (network, opt_state, log) where log is a dict of native floats.
    Timings are measured but presentation lives in the caller (fit).
    """

    # --- Init ---
    key_roll, key_update = jax.random.split(key)
    log = {}

    # --- Data collection (self-play) ---
    tic_roll = time.perf_counter()
    data_roll = collect(
        env=env, network=network, recurrent_fn=recurrent_fn, key=key_roll,
        n_roll=n_roll, n_batch=n_batch, n_step=n_step, num_sim=num_sim,
        discount=discount,
        temperature=temperature,
        dirichlet_fraction=dirichlet_fraction,
        dirichlet_alpha=dirichlet_alpha,
    )

    # --- Save logs ---
    # NB: mean on GPU (<...>.mean()) and put on RAM (float(...))
    time_roll = time.perf_counter() - tic_roll
    log["time_roll"] = time_roll
    log["value_mean"] = float(data_roll["value"].mean()) 

    # --- Mini-batch setup ---
    N = data_roll["value"].shape[0]
    size_minibatch = N // n_minibatch

    # --- Neural network updates ---
    # ToDo: jittable the NN update: jax.lax.scan for both loop
    tic_updates = time.perf_counter()
    logs_nn = []
    for idx_update in range(n_update):
        subkey_update = jax.random.fold_in(key_update, idx_update)
        idx_perm = jax.random.permutation(subkey_update, N)
        idx_perm_cut = idx_perm[:n_minibatch * size_minibatch]
        tic_update = time.perf_counter()

        mb_metrics = []
        for idx_mb in idx_perm_cut.reshape(n_minibatch, size_minibatch):
            data_minibatch = {k: v[idx_mb] for k, v in data_roll.items()}
            network, opt_state, loss = update_step(
                network=network,
                opt_state=opt_state,
                data_minibatch=data_minibatch,
                optimizer=optimizer,
                c_value=c_value,
            )
            mb_metrics.append(jax.device_get(loss))

        # --- Save update logs ---
        # NB: mean on GPU (<...>.mean()) and put on RAM (float(...))
        time_update = time.perf_counter() - tic_update
        mean_loss = {k: float(np.mean([m[k] for m in mb_metrics])) for k in mb_metrics[0]}
        logs_nn.append({
            "idx_update": idx_update,
            "time_update": time_update,
            **mean_loss,
        })

    log["time_updates"] = time.perf_counter() - tic_updates
    log["logs_nn"] = logs_nn

    return network, opt_state, log



