"""
ToDo: new comment header version

mctx recurrent_fn builder for a pgx environment (AlphaZero-style).

recurrent_fn = combined dynamics + evaluation, the only AZ/MuZero-swappable layer.
  dynamics   : pgx env.step (real simulator)
  evaluation : Equinox network -> (prior_logits, value)

Network flows through mctx's `params` channel (grad-ready, no recompile on weight updates).
Embedding carried in the tree = full pgx State.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import mctx


def make_recurrent_fn(env, discount: float = 0.99):
    # Batch pgx dynamics
    step = jax.vmap(env.step)

    def recurrent_fn(params, rng_key, action, embedding):
        network = params
        state = embedding
        _B = action.shape[0]

        # --- pgx dynamics ---
        keys = jax.random.split(rng_key, _B)
        next_state = step(state, action, keys)  # handles _step_count, terminated, observe

        # --- network evaluation ---
        # NB: prior_logits is not probability, mctx normalize in the RecurrentFnOutput
        # /!\ To understand /!\
        prior_logits, value = eqx.filter_vmap(network)(next_state.observation)

        # --- pgx -> mctx translations ---
        reward = next_state.rewards[:, 0]
        terminated = next_state.terminated.astype(jnp.float32)

        # --- Output ---
        out = mctx.RecurrentFnOutput(
            reward       = reward,
            discount     = discount * (1.0 - terminated),
            prior_logits = jnp.where(next_state.legal_action_mask, prior_logits, -jnp.inf),
            # value        = value,
            value = jnp.where(next_state.terminated, 0.0, value),
        )
        return out, next_state

    return recurrent_fn
