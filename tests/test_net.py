"""
Test networks integrity on the toy environement `env_toy`
"""

import jax
import jax.numpy as jnp

B = 16

def test_single_forward(network, env_toy, key):
    num_actions = env_toy.num_actions
    # observation_shape = env_toy.observation_shape
    state = env_toy.init(key)
    prior_logits, value = network(state.observation)
    assert prior_logits.shape == (num_actions,)
    assert value.shape == ()
    assert jnp.isfinite(prior_logits).all()


def test_batched_forward(network, env_toy, key):
    num_actions = env_toy.num_actions
    # observation_shape = env_toy.observation_shape
    key_B = jax.random.split(key, B)
    init_B = jax.vmap(env_toy.init)
    state_B = init_B(key_B)
    prior_logits, value = jax.vmap(network)(state_B.observation)
    assert prior_logits.shape == (B, num_actions)
    assert value.shape == (B,)
