"""
AlphaZero training step (single gradient pass over a self-play batch).

Targets come from selfplay():
  observation [T,B,obs]  policy [T,B,A]  value [T,B]  mask [T,B]
Time/batch axes are flattened to N=T*B independent samples; padded steps
(mask=False, post-termination) are excluded via masked means.

Loss = policy_loss + c_value * value_loss
  policy_loss : cross-entropy(network logits, MCTS visit distribution)
  value_loss  : MSE(network value, discounted return)
"""

import jax.numpy as jnp
import equinox as eqx
import optax


def _mean_masked_loss(loss, mask):
    # Average on real sample where `mask[i]=True`
    # Padded steps does not contribute (loss = 0)
    mask = mask.astype(loss.dtype)
    return jnp.sum(loss * mask) / jnp.sum(mask)


@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(network, data_game, c_value):
    # --- Flatten [T,B,*] -> [T*B,*] ---
    obs = data_game["observation"].reshape(-1, *data_game["observation"].shape[2:])
    z_policy = data_game["policy"].reshape(-1, data_game["policy"].shape[-1])
    z_value = data_game["value"].reshape(-1)
    mask = data_game["mask"].reshape(-1)

    # --- Network prediction ---
    y_policy, y_value = eqx.filter_vmap(network)(obs)

    # --- Loss ---
    # Policy: cross-entropy
    cross_entropy = optax.softmax_cross_entropy(y_policy, z_policy)
    policy_loss = _mean_masked_loss(cross_entropy, mask)
    # Value: mean squared
    mean_square = (y_value - z_value) ** 2
    value_loss = _mean_masked_loss(mean_square, mask)

    loss = policy_loss + c_value * value_loss
    return loss, {"policy_loss": policy_loss, "value_loss": value_loss}


@eqx.filter_jit
def train_step(network, opt_state, data_game, optimizer, c_value=1.0):
    # one gradient step; returns updated network + optimizer state + metrics
    (loss, loss_dict), grads = loss_fn(network, data_game, c_value)
    updates, opt_state = optimizer.update(grads, opt_state, network)
    network = eqx.apply_updates(network, updates)
    return network, opt_state, {"loss": loss, **loss_dict}


# ---- one self-play round, aggregating n_gen calls ----
def collect(env, network, recurrent_fn, key, B, n_gen, num_sim):
    games = []
    for g in range(n_gen):
        key, sub = split(key)
        games.append(selfplay(env, network, recurrent_fn, sub, B, num_sim))
    # each game has its own T -> pad all to T_max, concat on batch axis
    T_max = max(g["value"].shape[0] for g in games)
    games = [pad_time(g, T_max) for g in games]        # pad [T,B,*]->[T_max,B,*], mask=False on pad
    return concat_on_batch(games)                       # [T_max, n_gen*B, *]

# ---- flatten [T,B,*] -> [N,*] once per round ----
def flatten(data):
    N = T * B
    return {
        "observation": data["observation"].reshape(N, *obs_shape),
        "policy":      data["policy"].reshape(N, A),
        "value":       data["value"].reshape(N),
        "mask":        data["mask"].reshape(N),
    }

# ---- main AlphaZero loop ----
for it in range(n_iter):
    key, k_gen = split(key)
    data = collect(env, network, recurrent_fn, k_gen, B, n_gen, num_sim)  # regen with current net
    flat = flatten(data)                                                  # N = T_max * n_gen * B

    for update in range(n_update):
        key, k_perm = split(key)
        perm = permutation(k_perm, N)                    # full shuffle, each sample seen once
        for mb in split(perm, n_minibatch):              # n_minibatch slices of perm
            batch = index(flat, mb)                      # gather [mb_size, *]
            network, opt_state, m = train_step(network, opt_state, batch, optimizer, c_value)

    # (optional) eval / log every few iters
