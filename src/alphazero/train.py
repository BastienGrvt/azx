"""
ToDo: comment header


"""

from functools import partial
import jax
import equinox as eqx
import optax
import time

from .selfplay import selfplay
from .update import update_iter



def make_collect_global(
    gpu_mesh, gpu_axis,
    env, recurrent_fn, search_fn,
    n_roll, n_batch_local, n_step, discount,
):
    """
    Build the per-device data collector: n_roll rollouts on the local batch.
    NB: when sharded only array can be pass as argument
    """

    @partial(
        jax.shard_map, mesh=gpu_mesh,
        in_specs=({"network": jax.P(), "key": jax.P(gpu_axis)},),
        out_specs=jax.P(gpu_axis),
        check_vma=False,
    )
    def collect_local(args):
        network, key = args["network"], args["key"]
        """
        Data collector, does n_roll rollout
        1. Get data from n_roll call of selplay(...)
        2. Merge [n_roll, N, *] -> [N*n_roll, *] with N = B*n_step
        """
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

    return collect_local


def make_update_global(
    gpu_mesh, gpu_axis,
    optimizer, n_update, n_minibatch_local, size_minibatch_local, c_value,
):
    """
    Build the per-device update step: local shuffle, pmean'd gradients.
    NB: when sharded only array can be pass as argument
    """

    @partial(
        jax.shard_map, mesh=gpu_mesh,
        in_specs=({"network": jax.P(), "opt_state": jax.P(), "data_roll": jax.P(gpu_axis), "key": jax.P(gpu_axis)},),
        out_specs=(jax.P(), jax.P(), jax.P()),
        check_vma=False,
    )
    def update_local(args):
        key, network, data_roll, opt_state = args["key"], args["network"], args["data_roll"], args["opt_state"]
        """
        Per-device update: local shuffle, gradients pmean'd across devices
        """
        return update_iter(
            gpu_axis=gpu_axis,
            key=key[0], # NB: subkeys_update [n_gpu] -(sharp_map)-> key [1] x n_gpu (get key as key not array of key)
            network=network,
            data_roll=data_roll,
            opt_state=opt_state,
            optimizer=optimizer,
            c_value=c_value,
            n_update=n_update,
            n_minibatch=n_minibatch_local,
            size_minibatch=size_minibatch_local,
        )

    return update_local


def train_iter(
        gpu_mesh, gpu_axis,
        collect_global, update_global, 
        network, opt_state, key
    ):
    """
    Host-side wrapper: two sharded calls, timed separately.
    """
    # --- Init parallelization ---
    n_gpu = gpu_mesh.shape[gpu_axis]
    shard_keys = jax.NamedSharding(gpu_mesh, jax.P(gpu_axis))

    key_roll, key_update = jax.random.split(key)
    subkeys_roll = jax.device_put(jax.random.split(key_roll, n_gpu), shard_keys)
    subkeys_update = jax.device_put(jax.random.split(key_update, n_gpu), shard_keys)

    # --- Collect self-play data ---
    tic = time.perf_counter()
    data_roll = collect_global({
        "network": network,
        "key": subkeys_roll,
    })
    jax.block_until_ready(data_roll)
    tac_roll = time.perf_counter() - tic

    # --- Update ---
    tic = time.perf_counter()
    network, opt_state, logs_nn = update_global({
        "network": network,
        "opt_state": opt_state,
        "data_roll": data_roll,
        "key": subkeys_update,
    })
    jax.block_until_ready(logs_nn)
    tac_update = time.perf_counter() - tic

    # --- Metrics ---
    logs_nn = jax.device_get(logs_nn)
    log = {
        "time_roll": tac_roll,
        "time_updates": tac_update,
        "value_mean": float(data_roll["value"].mean()),
        "logs_nn": {k: v.tolist() for k, v in logs_nn.items()},
        "logs_nn_mean": {k: float(v.mean()) for k, v in logs_nn.items()},
    }
    return network, opt_state, log

