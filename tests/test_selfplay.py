"""
ToDo: Understand (from LLM, may be EnvToy specific regarding reward)

Test selfplay pipeline on the toy environement and network `env_toy` and `net_toy`

Refactored API (continuous rollout): selfplay returns FLAT [N, *] arrays with
N = n_step * B, no mask (every transition is valid), no time axis.
"""

import jax.numpy as jnp

from alphazero import make_recurrent_fn, selfplay

DISCOUNT = 0.9
B = 4
NUM_SIM = 8
N_STEP = 16


def test_selfplay(env_toy, net_toy, key):
    num_actions = env_toy.num_actions
    observation_shape = env_toy.observation_shape
    recurrent_fn = make_recurrent_fn(env_toy, discount=DISCOUNT)
    traj = selfplay(
        env_toy, net_toy, recurrent_fn, key,
        batch_size=B, num_simulations=NUM_SIM, n_step=N_STEP, discount=DISCOUNT,
    )

    N = N_STEP * B

    # --- shapes (flat, no time/batch split) ---
    assert traj["observation"].shape == (N, *observation_shape)
    assert traj["policy"].shape == (N, num_actions)
    assert traj["value"].shape == (N,)

    # --- policy rows are distributions over actions ---
    assert jnp.allclose(traj["policy"].sum(axis=-1), 1.0, atol=1e-5)
    assert (traj["policy"] >= 0).all()

    # --- value target bounded by a single game's return (no leak across resets) ---
    # reward is +1 at most once per game (terminal), so z = discount^k <= 1
    assert traj["value"].max() <= 1.0 + 1e-5
    assert traj["value"].min() >= 0.0 - 1e-5
