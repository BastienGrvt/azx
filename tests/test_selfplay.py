"""
Test selfplay pipeline on the toy environement and network `env_toy` and `net_toy`
"""

import jax.numpy as jnp

from alphazero import make_recurrent_fn, selfplay

DISCOUNT = 0.99
B = 4
NUM_SIM = 8

def test_selfplay(env_toy, net_toy, key):
    num_actions = env_toy.num_actions
    observation_shape = env_toy.observation_shape
    recurrent_fn = make_recurrent_fn(env_toy, discount=DISCOUNT)
    traj = selfplay(
            env_toy, net_toy, recurrent_fn, key,
            batch_size=B, num_simulations=NUM_SIM, discount=DISCOUNT
        )

    T, _ = traj["value"].shape
    assert traj["observation"].shape == (T, B, *observation_shape)
    assert traj["policy"].shape == (T, B, num_actions)
    assert traj["mask"].dtype == jnp.bool_
    # policy rows are distributions over actions
    assert jnp.allclose(traj["policy"].sum(axis=-1), 1.0, atol=1e-5)
    # mask is monotone: once a game is done it never comes back
    diffs = traj["mask"].astype(jnp.int32)[1:] - traj["mask"].astype(jnp.int32)[:-1]
    assert (diffs <= 0).all()                   # only True->False transitions
