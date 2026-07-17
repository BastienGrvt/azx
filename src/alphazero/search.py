"""
ToDo: new comment header

"""

import jax
import jax.numpy as jnp
import equinox as eqx
import mctx


def make_recurrent_fn(env, discount):
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

def search(
    network, recurrent_fn, state, key, num_sim,
    temperature, dirichlet_fraction, dirichlet_alpha,
):
    # Root prior + value from the network, mask illegal actions to -inf
    prior_logits, value = eqx.filter_vmap(network)(state.observation)
    prior_logits = jnp.where(state.legal_action_mask, prior_logits, -jnp.inf)

    # Root node
    root = mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        embedding=state,
    )

    # MCTS search
    return mctx.muzero_policy(
        params=network,
        rng_key=key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_sim,
        invalid_actions=~state.legal_action_mask,
        temperature=temperature,
        dirichlet_fraction=dirichlet_fraction,
        dirichlet_alpha=dirichlet_alpha,
    )
