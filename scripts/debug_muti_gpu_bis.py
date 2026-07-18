"""
Trancher: le surcout de shard_map est-il un cout FIXE par appel, ou un probleme
de memoire / taille de donnees ?

Les HLO sont identiques (2470 vs 2459 lignes, 51 fusions, 0 collectif) mais
l'execution est 25x plus lente -> ce n'est pas la compilation. Deux hypotheses:

  H1 - cout fixe par appel de shard_map (entree/sortie de region Manual,
       synchronisation host/device, transferts)
       => le surcout absolu reste constant quand n_batch diminue
       => le ratio EXPLOSE sur les petites tailles

  H2 - pression memoire (4 Go de VRAM, buffers intermediaires du lax.scan)
       => le surcout diminue avec n_batch
       => le ratio s'effondre vers 1x sur les petites tailles

Puis un profil JAX sur le cas le plus net (update) pour voir ou passe le temps.
"""

import time
import statistics

import jax
import equinox as eqx
import optax

from env import EnvToy
from net import NetToy
from alphazero import selfplay, update_iter, make_recurrent_fn, make_search_fn


N_REPEAT = 5
PROFILE_DIR = "/tmp/jax-trace"

# Fixed hparams (only n_batch varies)
N_ROLL, N_STEP, NUM_SIM, DISCOUNT = 4, 32, 16, 0.9
N_UPDATE, N_MINIBATCH, C_VALUE = 8, 16, 1.0
LR = 2e-2


def bench(fn, *args, n=N_REPEAT):
    """Median wall time of fn(*args), compilation excluded."""
    out = fn(*args)
    jax.block_until_ready(out)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


# ============================================================
# Setup
# ============================================================
env = EnvToy()
n_gpu = jax.device_count()
gpu_mesh = jax.make_mesh((n_gpu,), ("batch",),
                         axis_types=(jax.sharding.AxisType.Auto,))
jax.set_mesh(gpu_mesh)
gpu_axis = gpu_mesh.axis_names[0]
shard_keys = jax.NamedSharding(gpu_mesh, jax.P(gpu_axis))

network = NetToy(env.observation_shape, env.num_actions, key=jax.random.PRNGKey(0))
optimizer = optax.adam(LR)
opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
recurrent_fn = make_recurrent_fn(env, discount=DISCOUNT)
search_fn = make_search_fn(kind="muzero", num_sim=NUM_SIM, temperature=1.0,
                           dirichlet_fraction=0.25, dirichlet_alpha=0.3)

print(f"devices: {n_gpu} x {jax.devices()[0].device_kind}")
print(f"n_roll={N_ROLL}  n_step={N_STEP}  num_sim={NUM_SIM}\n")


def build_update_global(n_minibatch_local, size_minibatch_local):
    """shard_map envelope around update_iter, for a given data size."""
    from functools import partial

    @partial(
        jax.shard_map, mesh=gpu_mesh,
        in_specs=({"network": jax.P(), "opt_state": jax.P(),
                   "data_roll": jax.P(gpu_axis), "key": jax.P(gpu_axis)},),
        out_specs=(jax.P(), jax.P(), jax.P()),
        check_vma=False,
    )
    def update_local(args):
        return update_iter(
            gpu_axis=gpu_axis, key=args["key"][0],
            network=args["network"], data_roll=args["data_roll"],
            opt_state=args["opt_state"], optimizer=optimizer, c_value=C_VALUE,
            n_update=N_UPDATE, n_minibatch=n_minibatch_local,
            size_minibatch=size_minibatch_local,
        )

    return update_local


# ============================================================
# TEST 1: size sweep -- fixed cost or memory pressure ?
# ============================================================
print("=" * 78)
print("TEST 1  size sweep on UPDATE  (bare vs shard_map)")
print("=" * 78)
print(f"{'n_batch':>8} {'N':>10} {'bare':>9} {'shard_map':>11} {'ratio':>8} {'overhead':>10}")
print("-" * 78)

results = []
for n_batch in (256, 512, 1024, 2048, 4096):
    if n_batch % n_gpu:
        continue
    n_batch_local = n_batch // n_gpu
    n_minibatch_local = max(1, N_MINIBATCH // n_gpu)
    N_local = N_ROLL * N_STEP * n_batch_local
    size_minibatch_local = N_local // n_minibatch_local

    # Fake data of the right shape (no need to actually self-play here)
    k = jax.random.PRNGKey(1)
    data_roll = {
        "observation": jax.random.normal(k, (N_local, *env.observation_shape)),
        "policy": jax.random.uniform(k, (N_local, env.num_actions)),
        "value": jax.random.normal(k, (N_local,)),
    }
    data_roll = jax.device_put(data_roll, shard_keys)

    key_u = jax.random.PRNGKey(2)
    keys_u = jax.device_put(jax.random.split(key_u, n_gpu), shard_keys)

    bare = eqx.filter_jit(lambda net, opt, d, kk: update_iter(
        gpu_axis=None, key=kk, network=net, data_roll=d, opt_state=opt,
        optimizer=optimizer, c_value=C_VALUE, n_update=N_UPDATE,
        n_minibatch=n_minibatch_local, size_minibatch=size_minibatch_local,
    ))
    sharded = build_update_global(n_minibatch_local, size_minibatch_local)

    t_bare = bench(bare, network, opt_state, data_roll, key_u)
    t_shard = bench(sharded, {"network": network, "opt_state": opt_state,
                              "data_roll": data_roll, "key": keys_u})
    ratio = t_shard / t_bare
    overhead = t_shard - t_bare
    results.append((n_batch, N_local, t_bare, t_shard, ratio, overhead))
    print(f"{n_batch:>8} {N_local:>10} {t_bare:>8.4f}s {t_shard:>10.4f}s "
          f"{ratio:>7.1f}x {overhead:>9.4f}s")

print("-" * 78)
if len(results) >= 2:
    ov_small, ov_large = results[0][5], results[-1][5]
    print(f"\noverhead {results[0][0]} -> {results[-1][0]}: "
          f"{ov_small:.4f}s -> {ov_large:.4f}s")
    if ov_large < ov_small * 2:
        print("  => FIXED COST per shard_map call (H1). Overhead barely grows")
        print("     with data size. Look at region entry/exit, host sync, layout.")
    else:
        print("  => scales with data size (H2). Memory pressure or extra copies;")
        print("     check VRAM with a smaller n_batch in the real trainer.")


# ============================================================
# TEST 2: is the cost in the CALL or in the COMPUTE ?
# ============================================================
print("\n" + "=" * 78)
print("TEST 2  minimal payload -- isolates pure call overhead")
print("=" * 78)

from functools import partial

@partial(jax.shard_map, mesh=gpu_mesh,
         in_specs=(jax.P(gpu_axis),), out_specs=jax.P(gpu_axis), check_vma=False)
def noop_sharded(x):
    return x * 2.0

noop_jit = eqx.filter_jit(lambda x: x * 2.0)

x = jax.device_put(jax.random.normal(jax.random.PRNGKey(3), (n_gpu * 1024,)),
                   shard_keys)
t_j = bench(noop_jit, x)
t_s = bench(noop_sharded, x)
print(f"  {'jit  (x*2)':<28} {t_j*1e3:8.3f} ms")
print(f"  {'shard_map (x*2)':<28} {t_s*1e3:8.3f} ms")
print(f"  {'-> pure call overhead':<28} {(t_s-t_j)*1e3:+8.3f} ms")
print("\n  If this is ~0 ms, the overhead is NOT the shard_map call itself")
print("  but something about how the real workload runs inside it.")


# ============================================================
# TEST 3: profile the worst case
# ============================================================
print("\n" + "=" * 78)
print("TEST 3  profiler trace on the full-size update")
print("=" * 78)

n_batch = 4096
n_batch_local = n_batch // n_gpu
n_minibatch_local = max(1, N_MINIBATCH // n_gpu)
N_local = N_ROLL * N_STEP * n_batch_local
size_minibatch_local = N_local // n_minibatch_local

k = jax.random.PRNGKey(1)
data_roll = jax.device_put({
    "observation": jax.random.normal(k, (N_local, *env.observation_shape)),
    "policy": jax.random.uniform(k, (N_local, env.num_actions)),
    "value": jax.random.normal(k, (N_local,)),
}, shard_keys)
keys_u = jax.device_put(jax.random.split(jax.random.PRNGKey(2), n_gpu), shard_keys)
sharded = build_update_global(n_minibatch_local, size_minibatch_local)

# warmup (compile)
jax.block_until_ready(sharded({"network": network, "opt_state": opt_state,
                               "data_roll": data_roll, "key": keys_u}))

with jax.profiler.trace(PROFILE_DIR):
    for _ in range(3):
        out = sharded({"network": network, "opt_state": opt_state,
                       "data_roll": data_roll, "key": keys_u})
        jax.block_until_ready(out)

print(f"  trace written to {PROFILE_DIR}")
print(f"\n  Inspect with:")
print(f"    uv run tensorboard --logdir={PROFILE_DIR}")
print(f"    then open the 'Trace Viewer' tab")
print(f"\n  What to look for:")
print(f"    - large gaps between GPU kernels  -> host/device sync stalls")
print(f"    - memcpy / transfer ops           -> data moving unexpectedly")
print(f"    - one kernel dominating           -> a real compute regression")
