"""
Evaluation metrics, computed from raw rollout sequences.
"""

import jax.numpy as jnp


def eval_metrics(reward, terminated, discount_score, n_step):
    """
    Per-game metrics from a rollout (with `training=False`)
    Per-game metrics: [n_step, n_batch] -> [n_batch]

    Input:
      reward         [n_step, n_batch] -> per-step reward, r_t = R(o_t, a_t) = R(o_{t+1})
      terminated     [n_step, n_batch] -> per-step terminal flag
      discount_score scalar       -> discount used for `score` (defaults to mcts.discount)
      n_step         static       -> rollout length, sets the discount horizon

    Output:
      length  [n_batch] -> first terminal step, -1 if never terminated
      score   [n_batch] -> discounted return over the rollout
      success [n_batch] -> the game reached a terminal state
    """
    success = terminated.any(axis=0)
    discounts = discount_score ** jnp.arange(n_step)
    score = (reward * discounts[:, None]).sum(axis=0)
    length = jnp.where(success, jnp.argmax(terminated, axis=0) + 1, -1)
    return length, score, success


def aggregate_eval(length, score, success):
    """
    Full-batch aggregation from multi-gpu
    Based on `eval_metrics` output

    Input:
      length  [n_gpu * n_batch] -> first terminal step, -1 if never terminated
      score   [n_gpu * n_batch] -> discounted return over the rollout
      success [n_gpu * n_batch] -> the game reached a terminal state
    """
    n_success = success.sum()
    return {
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
