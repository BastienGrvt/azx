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
