"""
ToDo: comment header


"""

import jax
import numpy as np
import equinox as eqx
import optax
import time

from .selfplay import selfplay
from .update import update_iter




# ---- n_roll selfplay rounds, agregating data ----
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
    discount, c_value, temperature, dirichlet_fraction, dirichlet_alpha,
):
    key_roll, key_update = jax.random.split(key)
    log = {}

    # --- Data collection ---
    tic_roll = time.perf_counter()
    data_roll = collect(
        env=env, network=network,
        recurrent_fn=recurrent_fn,
        key=key_roll,
        n_roll=n_roll,
        n_batch=n_batch,
        n_step=n_step,
        num_sim=num_sim,
        discount=discount,
        temperature=temperature,
        dirichlet_fraction=dirichlet_fraction,
        dirichlet_alpha=dirichlet_alpha,
    )
    jax.block_until_ready(data_roll)  # accurate roll timing
    tac_roll = time.perf_counter() - tic_roll

    # --- Mini-batch setup ---
    N = data_roll["value"].shape[0]
    size_minibatch = N // n_minibatch

    # --- Neural network updates (single jitted call) ---
    tic_updates = time.perf_counter()
    network, opt_state, logs_nn = update_iter(
        network, opt_state, data_roll, optimizer, key_update,
        n_update, n_minibatch, size_minibatch, c_value,
    )
    jax.block_until_ready(logs_nn)
    logs_nn = jax.device_get(logs_nn)  
    tac_updates = time.perf_counter() - tic_updates

    # --- Save log of the iteration ---
    log["time_roll"] = tac_roll
    log["value_mean"] = float(data_roll["value"].mean())
    log["time_updates"] = tac_updates
    log["logs_nn"] = {k: v.tolist() for k, v in logs_nn.items()}
    log["logs_nn_mean"] = {k: float(v.mean()) for k, v in logs_nn.items()}
    return network, opt_state, log



