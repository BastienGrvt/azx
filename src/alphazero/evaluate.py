
import jax
import jax.numpy as jnp
import equinox as eqx
import mctx
from pgx.experimental import auto_reset
from .search import search




@eqx.filter_jit
def _rollout(env, network, recurrent_fn, key, n_batch, n_step, num_sim, discount):
    init = jax.vmap(env.init)
    step = jax.vmap(auto_reset(env.step, env.init))

    key, sub = jax.random.split(key)
    keys = jax.random.split(sub, n_batch)
    state = init(keys)

    def body_game(carry, _):
        state, key = carry
        key, key_mcts, key_step = jax.random.split(key, 3)

        # MCTS Search Call
        mcts_out = search(network, recurrent_fn, state, key_mcts, num_sim)

        # Pre-collection data (aligned on o_t)
        action = argmax(mcts_out.action_weights, axis=-1)

        # Play the move (auto_reset: on termination, slot restarts; reward/terminated preserved)
        state = step(state, mcts_out.action, jax.random.split(key_step, n_batch))

        # Post-collection data (reward of the action taken on o_t; terminated = cut flag)
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
