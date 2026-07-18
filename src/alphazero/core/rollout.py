"""
Game rollout and everything derived from it (AlphaZero).

rollout()   -> raw per-step sequences [n_step, n_batch]
targets()   -> value targets z, backward discounted return
flatten()   -> [n_step, n_batch, *] -> [n_step*n_batch, *]
selfplay()  -> rollout + targets + flatten, the training data producer

NB: MuZero will need its own `rollout_muzero` (learned model unroll).
"""

import jax
import jax.numpy as jnp
from pgx.experimental import auto_reset as auto_reset_wrapper


def rollout(
    env, network, recurrent_fn, search_fn, key,
    n_batch, n_step,
    training,
):
    """
    Play n_step moves on a batch of n_batch games.

    training=True  (selfplay): continuous rollout, terminal slots refilled with a
                   fresh game, stacks (observation, policy) for the training targets
    training=False (evaluate): one game per slot, terminal state frozen with zeroed
                   rewards, keeps only (reward, terminated) -> no [n_step, n_batch, obs] buffer
    """
    init = jax.vmap(env.init)
    step_fn = auto_reset_wrapper(env.step, env.init) if training else env.step
    step = jax.vmap(step_fn)

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
        # Training only: stacking these in eval would allocate a useless [n_step, n_batch, *] buffer
        pre_out = (state.observation, mcts_out.action_weights) if training else ()

        # Play the move
        state = step(state, mcts_out.action, jax.random.split(key_step, n_batch))

        # Post-collection data (reward and terminated: r_t = R(o_t, a_t) = R(o_{t+1}))
        rew_out, term_out = state.rewards[:, 0], state.terminated
        return (state, key), (*pre_out, rew_out, term_out)

    # --- Jax Scan ---
    # body(carry, X) -> new_carry, Y
    # lax.scan scans body for `length=n_step`
    # lax.scan stacks the Ys as (n_step, B, length(Y))
    # final_carry, Ys = lax.scan(body, init_carry, Xs, length=N)
    _, out = jax.lax.scan(body_game, (state, key), xs=None, length=n_step)

    # `training` is static: the output dict shape is fixed at trace time
    if training:
        obs, pi, rew, term = out
        return {"observation": obs, "policy": pi, "reward": rew, "terminated": term}
    rew, term = out
    return {"reward": rew, "terminated": term}


def targets(reward, terminated, discount):
    """
    Value targets: backward discounted return, cut at game boundaries via (1 - term).
    NB: assumes a continuous rollout (training=True), boundaries carried by `terminated`.
    """
    def body_value(carry, x):
        r, term = x
        # if new game: carry = r (last reward of the new game)
        # else: standard back-calculation
        carry = r + discount * carry * (1 - term)
        return carry, carry

    n_batch = reward.shape[1]
    _, val_rev = jax.lax.scan(body_value, jnp.zeros(n_batch), (reward[::-1], terminated[::-1]))
    return val_rev[::-1]


def flatten(observation, policy, value):
    """
    [n_step, n_batch, *] -> [n_step*n_batch, *]
    Plain reshape (no masking): jittable.
    """
    T, B = value.shape
    return {
        "observation": observation.reshape(T * B, *observation.shape[2:]),
        "policy":      policy.reshape(T * B, policy.shape[-1]),
        "value":       value.reshape(T * B),
    }


def selfplay(
    env, network, recurrent_fn, search_fn, key,
    n_batch, n_step,
    discount,
):
    """
    Training data producer: continuous rollout -> value targets -> flat batch.
    """
    data = rollout(
        env=env,
        network=network,
        recurrent_fn=recurrent_fn,
        search_fn=search_fn,
        key=key,
        n_batch=n_batch,
        n_step=n_step,
        training=True,
    )
    value = targets(
        reward=data["reward"],
        terminated=data["terminated"],
        discount=discount,
    )
    return flatten(
        observation=data["observation"],
        policy=data["policy"],
        value=value,
    )
