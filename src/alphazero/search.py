"""
ToDo: new comment header version

MCTS search: the root-eval + tree-search core shared by self-play and evaluation.

The only difference between training and eval is what the CALLER does with the
output (sample vs argmax, collect targets vs accumulate reward). The search
itself -- root network eval, illegal masking, mctx tree search -- lives here,
written once.

state -> network(obs) -> (prior, value) -> mctx.muzero_policy -> PolicyOutput
The returned PolicyOutput carries both:
  .action          sampled action (exploration, used by self-play)
  .action_weights  visit distribution (policy target; argmax for greedy eval)
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

def search(network, recurrent_fn, state, key, num_simulations):
    prior_logits, value = eqx.filter_vmap(network)(state.observation)
    prior_logits = jnp.where(state.legal_action_mask, prior_logits, -jnp.inf)

    # Root node
    root = mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        embedding=state,
    )
    
    # MCTS Search
    return mctx.muzero_policy(
        params=network,
        rng_key=key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=~state.legal_action_mask,
    )
