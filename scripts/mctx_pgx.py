"""
Minimal mctx search wired to a pgx environment.
Embedding carried in the tree = full pgx State (handles _step_count, terminated).
"""

import jax
import jax.numpy as jnp
import mctx

from env import EnvToy

# Batch
B = 10

# Search hyper-parameters
NUM_SIM = 2**6
DISCOUNT = 0.9

# Fake prior/value (no network yet)
FAKE_VALUE = 0.0
FAKE_PRIOR = 1.0


def _masked_prior(mask):
    # Set -inf reward for illegal mask
    logits = jnp.full(mask.shape, FAKE_PRIOR)
    return jnp.where(mask, logits, -jnp.inf)


def recurrent_fn(params, rng_key, action, embedding):
    _B = action.shape[0]
    keys = jax.random.split(rng_key, _B)

    # Pgx game dynamic
    new_state = step(embedding, action, keys)
    reward = new_state.rewards[:, 0] # [B, 1] -> [B]
    is_terminated = 1.0 - new_state.terminated.astype(jnp.float32)

    # Fake network
    prior_logits = _masked_prior(new_state.legal_action_mask)
    value        = jnp.full(_B, FAKE_VALUE)

    out = mctx.RecurrentFnOutput(
        reward       = reward,
        discount     = DISCOUNT * is_terminated,
        prior_logits = prior_logits,
        value        = value
    )
    return out, new_state


### GAME ###

env = EnvToy()
init = jax.vmap(env.init)
step = jax.vmap(env.step)

# Root
key = jax.random.PRNGKey(0)
init_state = init(jax.random.split(key, B))

root = mctx.RootFnOutput(
    prior_logits = _masked_prior(init_state.legal_action_mask),
    value        = jnp.full(B, FAKE_VALUE),
    embedding    = init_state,
)

policy_output = mctx.muzero_policy(
    params          = (),
    rng_key         = key,
    root            = root,
    recurrent_fn    = recurrent_fn,
    num_simulations = NUM_SIM,
    invalid_actions = ~init_state.legal_action_mask,   # mask at root
)

summary = policy_output.search_tree.summary()
print(summary.visit_probs)
