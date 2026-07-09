"""
Minimal AlphaZero network (Equinox): pgx State -> (prior_logits, value).

Single-state (no batch dim) — batching is done outside via vmap.
Shared MLP trunk + two heads: policy (logits over actions) and value (scalar).
"""

import jax
import math
import jax.numpy as jnp
import equinox as eqx


class NetToy(eqx.Module):
    trunk: eqx.nn.Linear
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear

    def __init__(self, observation_shape, num_actions, hidden=64, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        # Common network
        obs_dim = math.prod(observation_shape)
        self.trunk = eqx.nn.Linear(obs_dim, hidden, key=k1)
        # Heads
        self.policy_head = eqx.nn.Linear(hidden, num_actions, key=k2)
        self.value_head = eqx.nn.Linear(hidden, 1, key=k3)

    def __call__(self, obs):
        # Flatten observation to 1D vector (obs_dim,)
        x = jnp.ravel(obs)
        # Common trunk with relu activation
        h = jax.nn.relu(self.trunk(x))
        # Policy
        prior_logits = self.policy_head(h)
        # Value
        value = self.value_head(h)[0]
        return prior_logits, value
