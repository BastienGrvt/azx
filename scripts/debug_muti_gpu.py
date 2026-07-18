"""
Diagnose the post-refactor slowdown on a single device.

Compares, on identical data and identical parameters:
  A. train rollout : selfplay bare     vs  collect_global (shard_map)
  B. update        : update_iter bare  vs  update_global (shard_map)
  C. eval rollout  : _rollout bare     vs  evaluate_global (shard_map)
  D. pipelining    : cost of the intermediate block_until_ready

All timings are taken AFTER a warmup call, so compilation is excluded.

NB: dict keys differ between builders in the current repo --
    collect_global / update_global use "key", evaluate_global uses "keys".
"""

import time
import statistics

import jax
import equinox as eqx

from env import EnvToy
from net import NetToy
from alphazero import (
    Trainer, ConfigTrain, ConfigEval, MctsEngine, Meta,
    selfplay, update_iter,
)
from alphazero.evaluate import _rollout as eval_rollout


N_REPEAT = 5


def bench(fn, *args, n=N_REPEAT, label=""):
    """Time fn(*args) after a warmup call. Returns median."""
    out = fn(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t)
    med, best = statistics.median(ts), min(ts)
    print(f"  {label:<32} median {med:7.3f}s   min {best:7.3f}s")
    return med


def hlo_stats(lowered, label=""):
    """Compiled-HLO complexity: line count, fusions, collectives."""
    txt = lowered.compile().as_text()
    lines = len(txt.splitlines())
    fusions = txt.count("fusion(")
    coll = sum(txt.count(op) for op in
               ("all-reduce", "all-gather", "all-to-all", "collective-permute"))
    print(f"  {label:<32} lines {lines:6d}   fusions {fusions:5d}   collectives {coll:3d}")


# ============================================================
# Setup: same config as overfit_trainer.py
# ============================================================
def make_network(env, key):
    return NetToy(env.observation_shape, env.num_actions, key=key)

my_env = EnvToy()

config_train = ConfigTrain(
    mcts=MctsEngine(discount=0.9, num_sim=16, kind="muzero", temperature=1.0,
                    dirichlet_fraction=0.25, dirichlet_alpha=0.3,
                    pb_c_init=1.25, pb_c_base=800),
    optimizer="adam", optimizer_kwargs={"learning_rate": 2e-2},
    seed=0, n_iter=1, n_roll=4, n_batch=2**12, n_step=32,
    n_update=8, n_minibatch=16, c_value=1.0,
)
config_eval = ConfigEval(
    mcts=MctsEngine(discount=1.0, num_sim=16, kind="muzero",
                    temperature=0.0, dirichlet_fraction=0.0),
    seed=0, n_batch=2**12, n_step=32, discount_score=1.0,
)

trainer = Trainer(
    meta=Meta(env_id=my_env.id, net_id="NetToy", description="debug"),
    env=my_env, network_fn=make_network,
    config_train=config_train, config_eval=config_eval,
    dir_save="runs/_debug",
).init()

s = trainer.trainer_state
network, opt_state = s.network, s.opt_state
n_gpu, gpu_axis, gpu_mesh = trainer.n_gpu, trainer.gpu_axis, trainer.gpu_mesh
shard_keys = jax.NamedSharding(gpu_mesh, jax.P(gpu_axis))

print(f"devices: {n_gpu} x {jax.devices()[0].device_kind}")
print(f"n_batch={config_train.n_batch}  n_roll={config_train.n_roll}  "
      f"n_step={config_train.n_step}  num_sim={config_train.mcts.num_sim}\n")

key = jax.random.PRNGKey(0)
key_roll, key_update, key_eval = jax.random.split(key, 3)
subkeys_roll = jax.device_put(jax.random.split(key_roll, n_gpu), shard_keys)
subkeys_update = jax.device_put(jax.random.split(key_update, n_gpu), shard_keys)
subkeys_eval = jax.device_put(jax.random.split(key_eval, n_gpu), shard_keys)


# ============================================================
# A. TRAIN ROLLOUT
# ============================================================
print("=" * 74)
print("A. TRAIN ROLLOUT")
print("=" * 74)

selfplay_bare = eqx.filter_jit(
    lambda net, k: selfplay(
        env=my_env, network=net,
        recurrent_fn=trainer.recurrent_fn_train,
        search_fn=trainer.search_fn_train,
        key=k, n_batch=config_train.n_batch, n_step=config_train.n_step,
        discount=config_train.mcts.discount,
    )
)
med_bare = bench(selfplay_bare, network, key_roll, label="selfplay bare (1 roll)")

collect_call = lambda net, k: trainer.collect_global({"network": net, "key": k})
med_collect = bench(collect_call, network, subkeys_roll,
                    label=f"collect_global ({config_train.n_roll} rolls)")
per_roll = med_collect / config_train.n_roll
print(f"  {'-> per rollout':<32} {per_roll:7.3f}s"
      f"   ratio vs bare: {per_roll / med_bare:.2f}x")

print()
hlo_stats(jax.jit(selfplay_bare).lower(network, key_roll), "selfplay bare")
hlo_stats(jax.jit(collect_call).lower(network, subkeys_roll), "collect_global")


# ============================================================
# B. UPDATE
# ============================================================
print("\n" + "=" * 74)
print("B. UPDATE")
print("=" * 74)

data_roll = trainer.collect_global({"network": network, "key": subkeys_roll})
jax.block_until_ready(data_roll)
print(f"  data_roll: N={data_roll['value'].shape[0]}  "
      f"n_minibatch_local={trainer.n_minibatch_local}  "
      f"size_minibatch={trainer.size_minibatch_local}\n")

update_bare = eqx.filter_jit(
    lambda net, opt, d, k: update_iter(
        gpu_axis=None, key=k, network=net, data_roll=d, opt_state=opt,
        optimizer=trainer.optimizer, c_value=config_train.c_value,
        n_update=config_train.n_update,
        n_minibatch=trainer.n_minibatch_local,
        size_minibatch=trainer.size_minibatch_local,
    )
)
med_ubare = bench(update_bare, network, opt_state, data_roll, key_update,
                  label="update_iter bare")

update_call = lambda net, opt, d, k: trainer.update_global(
    {"network": net, "opt_state": opt, "data_roll": d, "key": k}
)
med_ushard = bench(update_call, network, opt_state, data_roll, subkeys_update,
                   label="update_global (shard_map)")
print(f"  {'-> ratio':<32} {med_ushard / med_ubare:.2f}x")

print()
hlo_stats(jax.jit(update_bare).lower(network, opt_state, data_roll, key_update),
          "update_iter bare")
hlo_stats(jax.jit(update_call).lower(network, opt_state, data_roll, subkeys_update),
          "update_global")


# ============================================================
# C. EVAL ROLLOUT  (worst observed factor: ~6x)
# ============================================================
print("\n" + "=" * 74)
print("C. EVAL ROLLOUT")
print("=" * 74)

eval_bare = eqx.filter_jit(
    lambda net, k: eval_rollout(
        env=my_env, network=net,
        recurrent_fn=trainer.recurrent_fn_eval,
        search_fn=trainer.search_fn_eval,
        key=k, n_batch=config_eval.n_batch, n_step=config_eval.n_step,
    )
)
med_ebare = bench(eval_bare, network, key_eval, label="_rollout bare")

eval_call = lambda net, k: trainer.evaluate_global({"network": net, "keys": k})
med_eshard = bench(eval_call, network, subkeys_eval, label="evaluate_global (shard_map)")
print(f"  {'-> ratio':<32} {med_eshard / med_ebare:.2f}x")

print()
hlo_stats(jax.jit(eval_bare).lower(network, key_eval), "_rollout bare")
hlo_stats(jax.jit(eval_call).lower(network, subkeys_eval), "evaluate_global")


# ============================================================
# D. PIPELINING
# ============================================================
print("\n" + "=" * 74)
print("D. PIPELINING (collect + update)")
print("=" * 74)

def chained_sync(net, opt, kr, ku):
    """Current train_iter shape: blocks between the two envelopes."""
    d = trainer.collect_global({"network": net, "key": kr})
    jax.block_until_ready(d)
    return trainer.update_global(
        {"network": net, "opt_state": opt, "data_roll": d, "key": ku}
    )

def chained_async(net, opt, kr, ku):
    """Same, without the intermediate barrier."""
    d = trainer.collect_global({"network": net, "key": kr})
    return trainer.update_global(
        {"network": net, "opt_state": opt, "data_roll": d, "key": ku}
    )

med_sync = bench(chained_sync, network, opt_state, subkeys_roll, subkeys_update,
                 label="collect+update (blocking)")
med_async = bench(chained_async, network, opt_state, subkeys_roll, subkeys_update,
                  label="collect+update (async)")
print(f"  {'-> barrier cost':<32} {med_sync - med_async:+7.3f}s")


# ============================================================
print("\n" + "=" * 74)
print("READING THE RESULTS")
print("=" * 74)
print("""
  ratio ~= 1x          -> shard_map is not the cost; look elsewhere
                          (train_iter structure, key handling, ...)
  ratio >> 1x          -> Manual mode blocks XLA optimisations; the dual-path
                          option (bypass shard_map when n_gpu == 1) is worth it
  fusions much lower   -> confirms lost fusion, same conclusion
  barrier cost > 0     -> merging the two envelopes into one would help
""")
