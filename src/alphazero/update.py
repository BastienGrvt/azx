"""
ToDo: new comment header version

"""

import jax
import numpy as np
import equinox as eqx
import optax
import time


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

@eqx.filter_jit
def update_iter(
    network, opt_state, data_roll, optimizer, key_update,
    n_update, n_minibatch, size_minibatch, c_value,
):
    # Two nested scans: 
    # -> outer over updates (reshuffle each time),
    # -> inner over minibatches (one grad step each)
    N = n_minibatch * size_minibatch
    subkeys_update = jax.random.split(key_update, n_update)

    def body_update(carry, subkey_update):
        network, opt_state = carry
        # Setup permutation indices
        idx_perm = jax.random.permutation(subkey_update, N)
        idx_mb_all = idx_perm.reshape(n_minibatch, size_minibatch)

        def body_minibatch(carry, idx_mb):
            network, opt_state = carry
            # Builde the shuffeled data from the permutation indices
            data_mb = {k: v[idx_mb] for k, v in data_roll.items()}
            network, opt_state, loss = update_step(network, opt_state, data_mb, optimizer, c_value)
            return (network, opt_state), loss

        # Scan over minibatches permutation indices and update neural network
        # idx_mb_all = [[1, 5, 2], [3, 6, 8], [4, 9, 7]]
        # idx_mb = [1, 5, 2]
        # data_roll[key]: [N, *] -> data_roll[key][idx_mb]: [3, *]
        (network, opt_state), losses = jax.lax.scan(body_minibatch, (network, opt_state), idx_mb_all)
        # Compute the mean loss over the minibatches
        mean_loss = {k: v.mean() for k, v in losses.items()}
        return (network, opt_state), mean_loss
    # Scan over subkeys_update (n_update random keys)
    (network, opt_state), logs_nn = jax.lax.scan(body_update, (network, opt_state), subkeys_update)
    # logs_nn: dict of arrays, each shape [n_update]
    return network, opt_state, logs_nn


