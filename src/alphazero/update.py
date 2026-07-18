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
    # batch already flat [minibatch_size, *] -> predict directly
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


def update_step(gpu_axis, network, opt_state, data_minibatch, optimizer, c_value):
    (loss, loss_dict), grads = loss_fn(network, data_minibatch, c_value)
    grads = jax.lax.pmean(grads, gpu_axis)
    loss = jax.lax.pmean(loss, gpu_axis)
    loss_dict = jax.lax.pmean(loss_dict, gpu_axis)
    updates, opt_state = optimizer.update(grads, opt_state, network)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, {"loss": loss, **loss_dict}



@eqx.filter_jit
def update_iter(
    gpu_axis,
    network, opt_state, data_roll, optimizer, key,
    n_update, n_minibatch, size_minibatch, c_value,
):
    # Two nested scans: 
    # -> outer over updates (reshuffle each time),
    # -> inner over minibatches (one grad step each)
    N = n_minibatch * size_minibatch
    subkeys_update = jax.random.split(key, n_update)

    def body_update(carry, subkey_update):
        network, opt_state = carry
        # Setup permutation indices
        idx_perm = jax.random.permutation(subkey_update, N)
        idx_minibatch_all = idx_perm.reshape(n_minibatch, size_minibatch)

        def body_minibatch(carry, idx_minibatch):
            network, opt_state = carry
            # Builde the shuffeled data from the permutation indices
            data_minibatch = {k: v[idx_minibatch] for k, v in data_roll.items()}
            network, opt_state, loss = update_step(
                    gpu_axis=gpu_axis,
                    network=network,
                    opt_state=opt_state,
                    data_minibatch=data_minibatch,
                    optimizer=optimizer,
                    c_value=c_value,
                )
            return (network, opt_state), loss

        # Scan over minibatches permutation indices and update neural network
        # idx_minibatch_all = [[1, 5, 2], [3, 6, 8], [4, 9, 7]]
        # idx_minibatch = [1, 5, 2]
        # data_roll[key]: [N, *] -> data_roll[key][idx_minibatch]: [3, *]
        (network, opt_state), losses = jax.lax.scan(body_minibatch, (network, opt_state), idx_minibatch_all)
        # Compute the mean loss over the minibatches
        mean_loss = {k: v.mean() for k, v in losses.items()}
        return (network, opt_state), mean_loss
    # Scan over subkeys_update (n_update random keys)
    (network, opt_state), logs_nn = jax.lax.scan(body_update, (network, opt_state), subkeys_update)
    # logs_nn: dict of arrays, each shape [n_update]
    return network, opt_state, logs_nn


