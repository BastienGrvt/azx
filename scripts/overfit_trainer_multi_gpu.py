"""
Minimal multi-device pipeline check on 4 simulated CPU devices.

A single physical GPU cannot be split into several JAX devices
(--xla_force_host_platform_device_count only applies to the CPU backend), so
this runs on CPU. It validates CORRECTNESS, never performance:

  1. the sharded pipeline compiles and runs end-to-end with n_gpu > 1
  2. network replicas stay bit-identical across devices
     (replaces check_vma, which mctx forces us to disable)
  3. the loss goes down and eval metrics stay in range

Sizes are tiny on purpose: CPU x4 is slow, we only care that it works.
"""

# Must be set before importing jax
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import jax
import numpy as np

from env import EnvToy
from net import NetToy
from alphazero import Trainer, ConfigTrain, ConfigEval, MctsEngine, Meta


def make_network(env, key):
    return NetToy(env.observation_shape, env.num_actions, key=key)


my_env = EnvToy()
n_gpu = jax.device_count()
print(f"devices: {n_gpu} x {jax.devices()[0].device_kind}\n")

config_train = ConfigTrain(
    mcts=MctsEngine(discount=0.9, num_sim=16, kind="muzero", temperature=1.0,
                    dirichlet_fraction=0.25, dirichlet_alpha=0.3),
    optimizer="adam", optimizer_kwargs={"learning_rate": 2e-2},
    seed=0,
    n_iter=5,
    n_roll=8,
    n_batch=64,        # 16 per device
    n_step=16,
    n_update=8,
    n_minibatch=8,     # 2 per device
    c_value=1.0,
)

config_eval = ConfigEval(
    mcts=MctsEngine(discount=1.0, num_sim=16, kind="muzero",
                    temperature=0.0, dirichlet_fraction=0.0),
    seed=0, n_batch=64, n_step=16, discount_score=1.0,
)

trainer = Trainer(
    meta=Meta(env_id=my_env.id, net_id="NetToy", description="multi-device check"),
    env=my_env, network_fn=make_network,
    config_train=config_train, config_eval=config_eval,
    dir_save="runs/_multi_gpu_check",
).init()

print(f"n_batch_local     = {trainer.n_batch_local_train}")
print(f"n_minibatch_local = {trainer.n_minibatch_local}")
print(f"size_minibatch    = {trainer.size_minibatch_local}\n")

t0 = time.perf_counter()
logs = trainer.fit(eval_every=1, checkpoint_every=None, verbose=True)
elapsed = time.perf_counter() - t0


# ===== CHECKS =====
print("=" * 60)

# 1. replicas must be bit-identical (pmean working)
shards = trainer.trainer_state.network.trunk.weight.addressable_shards
print(f"shards visible    : {len(shards)}")
if len(shards) > 1:
    ref = np.asarray(shards[0].data)
    diff = max(float(np.abs(np.asarray(s.data) - ref).max()) for s in shards[1:])
    print(f"replica max |diff|: {diff:.3e}   [{'OK' if diff == 0.0 else 'DIVERGED'}]")
    if diff:
        print("  -> pmean missing or misplaced in update_step")
else:
    print("replica check     : single shard, cannot compare")

# 2. learning
l0, l1 = logs[0]["logs_nn_mean"]["loss"], logs[-1]["logs_nn_mean"]["loss"]
print(f"loss              : {l0:.4f} -> {l1:.4f}   "
      f"[{'OK' if l1 < l0 else 'NOT LEARNING'}]")

# 3. eval metrics in range (catches a bad reduction)
evals = [l["eval"] for l in logs if l["eval"] is not None]
if evals:
    sane = all(0.0 <= e["success"] <= 1.0 for e in evals)
    print(f"success rate      : {evals[0]['success']:.2f} -> {evals[-1]['success']:.2f}   "
          f"[{'OK' if sane else 'OUT OF RANGE'}]")

print(f"\ndone in {elapsed:.1f}s")
print("=" * 60)
