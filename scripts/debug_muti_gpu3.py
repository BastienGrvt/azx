"""
Le surcout de shard_map vient-il d'une recompilation a chaque appel ?
La trace montre CompileSingleModule x203, PTX->CUBIN x86 -> oui.
Hypothese: shard_map seul ne cache pas la compilation, il faut un jit autour.
"""

import time, statistics
from functools import partial
import jax
import equinox as eqx

N_REPEAT = 10

def bench(fn, *args, n=N_REPEAT):
    out = fn(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        out = fn(*args); jax.block_until_ready(out)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)

n_gpu = jax.device_count()
gpu_mesh = jax.make_mesh((n_gpu,), ("batch",),
                         axis_types=(jax.sharding.AxisType.Auto,))
jax.set_mesh(gpu_mesh)
gpu_axis = gpu_mesh.axis_names[0]
shard = jax.NamedSharding(gpu_mesh, jax.P(gpu_axis))

x = jax.device_put(jax.random.normal(jax.random.PRNGKey(0), (n_gpu * 1024,)), shard)

print(f"devices: {n_gpu} x {jax.devices()[0].device_kind}\n")

# --- A: plain jit (reference) ---
f_jit = eqx.filter_jit(lambda v: v * 2.0)

# --- B: bare shard_map (current code) ---
f_shmap = jax.shard_map(
    lambda v: v * 2.0, mesh=gpu_mesh,
    in_specs=(jax.P(gpu_axis),), out_specs=jax.P(gpu_axis), check_vma=False,
)

# --- C: shard_map wrapped in jit (the fix, if the hypothesis holds) ---
f_shmap_jit = eqx.filter_jit(f_shmap)

t_a, t_b, t_c = bench(f_jit, x), bench(f_shmap, x), bench(f_shmap_jit, x)

print(f"  {'A. jit(x*2)':<34} {t_a*1e3:9.3f} ms")
print(f"  {'B. shard_map(x*2)':<34} {t_b*1e3:9.3f} ms   {t_b/t_a:7.1f}x")
print(f"  {'C. jit(shard_map(x*2))':<34} {t_c*1e3:9.3f} ms   {t_c/t_a:7.1f}x")

print()
if t_c < t_b / 5:
    print("  => CONFIRMED: bare shard_map recompiles on every call.")
    print("     Fix: wrap each envelope in eqx.filter_jit inside the builders,")
    print("     i.e. `return eqx.filter_jit(collect_local)`.")
else:
    print("  => NOT the cause: wrapping in jit does not help.")
    print("     The overhead is inherent to entering Manual mode; the dual-path")
    print("     option (bypass shard_map when n_gpu == 1) becomes the way out.")
