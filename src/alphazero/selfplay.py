"""
Self-play: fully-jittable continuous rollout via lax.scan on a fixed n_step.

Continuous rollout: env.step is wrapped with pgx auto_reset, so a slot that
terminates immediately restarts a fresh game in-place. No padding, no wasted
VRAM, every step yields a valid transition -> the whole rollout is jittable
(no data-dependent masking).

Per step (scan body):
  1. root eval: network(state) -> (prior, value), mask illegal actions
  2. mctx search -> improved policy (visit distribution)
  3. collect target BEFORE step: (obs, policy)
  4. play sampled action on the REAL env (auto_reset refills terminal slots)
  5. collect AFTER step: (reward, terminated) -- terminated is the cut flag

Value target = discounted return z_t = Σ γ^k r_{t+k}, computed via backward
scan. The return is CUT at game boundaries: z = r + γ·z·(1 - terminated), so a
game's return never leaks across the reset into the next game in the slot.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import mctx
from pgx.experimental import auto_reset
from .search import search


@eqx.filter_jit
def _rollout(env, network, recurrent_fn, key, n_batch, n_step, num_sim, discount):
    init = jax.vmap(env.init)
    step = jax.vmap(auto_reset(env.step, env.init))  # refills terminal slots

    key, sub = jax.random.split(key)
    keys = jax.random.split(sub, n_batch)
    state = init(keys)

    def body_game(carry, _):
        state, key = carry
        key, key_mcts, key_step = jax.random.split(key, 3)

        # MCTS Search Call
        mcts_out = search(network, recurrent_fn, state, key_mcts, num_sim)

        # Pre-collection data (aligned on o_t)
        obs_out, pi_out = state.observation, mcts_out.action_weights

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


@eqx.filter_jit
def selfplay(env, network, recurrent_fn, key, n_batch, n_step, num_sim, discount):
    data = _rollout(
            env=env, network=network, recurrent_fn=recurrent_fn, key=key, 
            n_batch=n_batch, n_step=n_step, num_sim=num_sim, 
            discount=discount
        )
    return _flatten(data)



