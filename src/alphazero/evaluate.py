import jax
import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
def _rollout(
    env, network, recurrent_fn, search_fn, key,
    n_batch, n_step,
    discount, discount_score,
):
    init = jax.vmap(env.init)
    step = jax.vmap(env.step)

    key, sub = jax.random.split(key)
    keys = jax.random.split(sub, n_batch)
    state = init(keys)

    def body_game(carry, _):
        state, key = carry
        key, key_mcts, key_step = jax.random.split(key, 3)

        # MCTS search
        mcts_out = search_fn(
            network=network,
            recurrent_fn=recurrent_fn,
            state=state,
            key=key_mcts,
        )

        # Greedy play
        state = step(state, mcts_out.action, jax.random.split(key_step, n_batch))

        # Get reward and terminated hook
        reward_out, terminated_out = state.rewards[:, 0], state.terminated
        return (state, key), (reward_out, terminated_out)

    # --- Play the games ---
    _, (reward, terminated) = jax.lax.scan(body_game, (state, key), xs=None, length=n_step)

    # --- Metrics ---
    # Games succeded
    success = terminated.any(axis=0)
    n_success = success.sum()
    # Games score: score = sum_k gama^k * reward
    discounts_score = discount_score ** jnp.arange(n_step)
    score = (reward * discounts_score[:, None]).sum(axis=0)
    # Length
    length = jnp.where(success, jnp.argmax(terminated, axis=0) + 1, -1)
    # Mean values
    mean_success = success.mean()
    mean_score  = score.mean()
    mean_length  = jnp.where(n_success > 0, (length * success).sum() / jnp.maximum(n_success, 1), 0.0)

    return {
        "length": length,
        "score": score,
        "success": success,
        "n_success": n_success,
        "mean_success": mean_success,
        "mean_score":  mean_score,
        "mean_length":  mean_length,
    }


@eqx.filter_jit
def evaluate(
    env, network, recurrent_fn, search_fn, key,
    n_batch, n_step,
    discount, discount_score,
):
    metrics = _rollout(
        env=env, network=network,
        recurrent_fn=recurrent_fn,
        search_fn=search_fn,
        key=key,
        n_batch=n_batch,
        n_step=n_step,
        discount=discount,
        discount_score=discount_score,
    )
    return metrics
