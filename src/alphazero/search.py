"""
ToDo: new comment header

"""

import jax
import jax.numpy as jnp
import equinox as eqx
import mctx

_POLICIES = {"muzero": mctx.muzero_policy, "gumbel": mctx.gumbel_muzero_policy}

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

def make_search_fn(kind, num_sim, **policy_kwargs):
    # Set the policy
    if kind not in _POLICIES:
        raise ValueError(f"unknown policy kind: {kind!r}, expected {list(_POLICIES)}")
    mctx_policy = _POLICIES[kind]

    def search_fn(network, recurrent_fn, state, key):
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
        return mctx_policy(
            params=network,
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_sim,
            invalid_actions=~state.legal_action_mask,
            **policy_kwargs
        )

    return search_fn


