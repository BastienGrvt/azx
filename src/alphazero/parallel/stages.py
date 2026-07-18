"""
Sharded stages: shard_map wrappers around the pure functions of `core/`

No algorithm, only the glue for shard_map
  - Setup what is replicated (network, opt_state) vs sharded (keys, data)
  - Setup local sizes from ShardingPlan

NB: shard_map only accepts arrays as arguments, we use dict-of-arrays convention for explicit arguments
NB: a sharded key comes in as an array of shape [1], hence the `key[0]`
"""

from functools import partial

import jax
import equinox as eqx

from ..core.rollout import selfplay, rollout
from ..core.loss import update_iter
from ..core.metrics import eval_metrics


def make_collect(plan, env, recurrent_fn, search_fn, config_train):
    """
    Build the per-device data collector: n_roll rollouts on the local batch
    """
    n_roll = config_train.n_roll
    n_step = config_train.n_step
    discount = config_train.mcts.discount
    n_batch_local = plan.n_batch_local_train

    @partial(
        jax.shard_map, mesh=plan.mesh,
        in_specs=({"network": plan.P_rep, "key": plan.P_shard},),
        out_specs=plan.P_shard,
        check_vma=False,
    )
    def collect_local(args):
        """
        Data collector, does n_roll rollout
        1. Get data from n_roll call of selfplay(...)
        2. Merge [n_roll, N, *] -> [N*n_roll, *] with N = B*n_step
        """
        network, key = args["network"], args["key"]

        def body_rollout(key, _):
            key, subkey_game = jax.random.split(key)
            game = selfplay(
                env=env,
                network=network,
                recurrent_fn=recurrent_fn,
                search_fn=search_fn,
                key=subkey_game,
                n_batch=n_batch_local,
                n_step=n_step,
                discount=discount,
            )
            return key, game

        _, data = jax.lax.scan(body_rollout, key[0], xs=None, length=n_roll)
        # [n_roll, n_step*n_batch_local, *] -> [N_local, *]
        return {k: v.reshape(-1, *v.shape[2:]) for k, v in data.items()}

    return eqx.filter_jit(collect_local)


def make_update(plan, optimizer, config_train):
    """
    Build the per-device update step: local shuffle, pmean'd gradients
    """
    n_update = config_train.n_update
    c_value = config_train.c_value
    n_minibatch_local = plan.n_minibatch_local
    size_minibatch_local = plan.size_minibatch_local
    gpu_axis = plan.axis

    @partial(
        jax.shard_map, mesh=plan.mesh,
        in_specs=({
            "network":   plan.P_rep,
            "opt_state": plan.P_rep,
            "data_roll": plan.P_shard,
            "key":       plan.P_shard,
        },),
        out_specs=(plan.P_rep, plan.P_rep, plan.P_rep),
        check_vma=False,
    )
    def update_local(args):
        """
        Per-device update: local shuffle, gradients pmean'd across devices
        """
        network, opt_state = args["network"], args["opt_state"]
        data_roll, key = args["data_roll"], args["key"]

        return update_iter(
            network=network,
            opt_state=opt_state,
            data_roll=data_roll,
            optimizer=optimizer,
            key=key[0],
            n_update=n_update,
            n_minibatch=n_minibatch_local,
            size_minibatch=size_minibatch_local,
            c_value=c_value,
            gpu_axis=gpu_axis,
        )

    return eqx.filter_jit(update_local)


def make_evaluate(plan, env, recurrent_fn, search_fn, config_eval):
    """
    Build the evaluator
    """
    n_step = config_eval.n_step
    discount_score = config_eval.discount_score
    n_batch_local = plan.n_batch_local_eval

    @partial(
        jax.shard_map, mesh=plan.mesh,
        in_specs=({"network": plan.P_rep, "key": plan.P_shard},),
        out_specs=(plan.P_shard, plan.P_shard, plan.P_shard),
        check_vma=False,
    )
    def evaluate_local(args):
        """
        Run local rollout
        Output per-game metrics
        """
        network, key = args["network"], args["key"]

        data = rollout(
            env=env,
            network=network,
            recurrent_fn=recurrent_fn,
            search_fn=search_fn,
            key=key[0],
            n_batch=n_batch_local,
            n_step=n_step,
            training=False,
        )
        return eval_metrics(
            reward=data["reward"],
            terminated=data["terminated"],
            discount_score=discount_score,
            n_step=n_step,
        )

    return eqx.filter_jit(evaluate_local)



