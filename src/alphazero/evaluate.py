import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial


# @eqx.filter_jit
def _rollout(env, network, recurrent_fn, search_fn, key, n_batch, n_step):
    """
    Greedy rollout
    Returns raw per-game data, no aggregation
    """
    init = jax.vmap(env.init)
    step = jax.vmap(env.step)

    key, sub = jax.random.split(key)
    keys = jax.random.split(sub, n_batch)
    state = init(keys)

    def body_game(carry, _):
        state, key = carry
        key, key_mcts, key_step = jax.random.split(key, 3)

        mcts_out = search_fn(
            network=network, recurrent_fn=recurrent_fn,
            state=state, key=key_mcts,
        )
        state = step(state, mcts_out.action, jax.random.split(key_step, n_batch))

        reward_out, terminated_out = state.rewards[:, 0], state.terminated
        return (state, key), (reward_out, terminated_out)

    _, (reward, terminated) = jax.lax.scan(body_game, (state, key), xs=None, length=n_step)
    return reward, terminated



def make_evaluate_global(
    gpu_mesh, gpu_axis,
    env, recurrent_fn, search_fn,
    n_batch_local, n_step, 
    discount_score,
):
    """
    Build the sharded evaluator
    NB: when sharded only array can be pass as argument
    """

    @partial(
        jax.shard_map, mesh=gpu_mesh,
        in_specs=({"network": jax.P(), "keys": jax.P(gpu_axis)},),
        out_specs=(jax.P(gpu_axis), jax.P(gpu_axis), jax.P(gpu_axis)),
        check_vma=False,
    )
    def evaluate_local(args):
        network, keys = args["network"], args["keys"]
        """
        Per-device game rollout
        Metrics are computed host-side in `evaluate()` on the full batch
        """
        reward, terminated = _rollout(
            env=env,
            network=network,
            recurrent_fn=recurrent_fn,
            search_fn=search_fn,
            key=keys[0],
            n_batch=n_batch_local,
            n_step=n_step,
        )
        success = terminated.any(axis=0)
        discounts = discount_score ** jnp.arange(n_step)
        score = (reward * discounts[:, None]).sum(axis=0)
        length = jnp.where(success, jnp.argmax(terminated, axis=0) + 1, -1)
        return length, score, success

    return eqx.filter_jit(evaluate_local)


def evaluate(gpu_mesh, gpu_axis, evaluate_global, network, key):
    """
    Host-side wrapper: sharded rollout, then metrics on the full batch.
    """
    # --- Initialisation ---
    n_gpu = gpu_mesh.shape[gpu_axis]
    shard_keys = jax.NamedSharding(gpu_mesh, jax.P(gpu_axis))
    keys = jax.device_put(jax.random.split(key, n_gpu), shard_keys)

    # --- Evaluation games ---
    length, score, success = evaluate_global({
            "network": network,
            "keys": keys,
        })

    # --- Full-batch metrics ---
    n_success = success.sum()
    metrics = {
        "length": length,
        "score": score,
        "success": success,
        "n_success": n_success,
        "mean_success": success.mean(),
        "mean_score": score.mean(),
        "mean_length": jnp.where(
            n_success > 0,
            (length * success).sum() / jnp.maximum(n_success, 1), 0.0
        ),
    }

    return metrics
