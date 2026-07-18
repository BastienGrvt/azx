"""
ToDo: Set proper header comment
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from pgx.experimental import auto_reset


# @eqx.filter_jit
def _rollout(
    env, network, recurrent_fn, search_fn, key, 
    n_batch, n_step,
    discount,
):
    init = jax.vmap(env.init)
    step = jax.vmap(auto_reset(env.step, env.init))  # refills terminal slots
    
    # Initialize the env
    key, key_sub = jax.random.split(key)
    keys_init = jax.random.split(key_sub, n_batch)
    state = init(keys_init)

    def body_game(carry, _):
        state, key = carry
        key, key_mcts, key_step = jax.random.split(key, 3)

        # MCTS Search Call
        mcts_out = search_fn(
            network=network, 
            recurrent_fn=recurrent_fn, 
            state=state, 
            key=key_mcts, 
        )

        # Pre-collection data (observation and policy: o_t, pi_t)
        obs_out, pi_out = state.observation, mcts_out.action_weights

        # Play the move
        state = step(state, mcts_out.action, jax.random.split(key_step, n_batch))

        # Post-collection data (reward and terminated: r_t = R(o_t, a_t)=R(o_{t+1}) )
        rew_out, term_out = state.rewards[:, 0], state.terminated
        return (state, key), (obs_out, pi_out, rew_out, term_out)

    # backward discounted return, cut at game boundaries via (1 - term)
    def body_value(carry, x):
        r, term = x
        # if new game: carry = r (last reward of the new game)
        # else: standard back-calculation
        carry = r + discount * carry * (1 - term)
        return carry, carry

    # --- Jax Scan ---

    # body(carry, X) -> new_carry, Y
    # lax.scan scans body for `length=n_step`
    # lax.scan stacks the Ys as (n_step, B, length(Y))
    # final_carry, Ys = lax.scan(body, init_carry, Xs, length=N)

    # Game
    _, (obs, pi, rew, term) = jax.lax.scan(body_game, (state, key), xs=None, length=n_step)
    # Value (backward, boundaries cut by term)
    _, val_rev = jax.lax.scan(body_value, jnp.zeros(n_batch), (rew[::-1], term[::-1]))
    val = val_rev[::-1]

    return {"observation": obs, "policy": pi, "value": val}


def _flatten(data):
    # [T, B, *] -> [T*B, *]
    # Now a plain reshape (no masking): jittable.
    T, B = data["value"].shape
    return {
        "observation": data["observation"].reshape(T * B, *data["observation"].shape[2:]),
        "policy":      data["policy"].reshape(T * B, data["policy"].shape[-1]),
        "value":       data["value"].reshape(T * B),
    }


# @eqx.filter_jit
def selfplay(
    env, network, recurrent_fn, search_fn, key, 
    n_batch, n_step,
    discount,
):
    data = _rollout(
            env=env,
            network=network,
            recurrent_fn=recurrent_fn,
            search_fn=search_fn,
            key=key,
            n_batch=n_batch,
            n_step=n_step,
            discount=discount, 
        )
    return _flatten(data)



