"""
Exemple minimal mctx : navigation 1D.
Benchmark: time + peak VRAM across batch sizes.

Measurment:

bastiengrvt@Bastien-PC:~/dev/proj/jax/test_mctx$ uv run minimal_2.py
Device: cuda:0

     B    time_ms    ms/elem    peak_MB
     1     88.406    88.4058      100.7
     2    127.080    63.5398      100.7
     4    126.256    31.5639      100.7
     8    129.433    16.1791      100.7
    16    205.289    12.8305      100.7
    32    201.653     6.3016      100.8
    64    204.232     3.1911      100.8
   128    226.330     1.7682      100.8
   256    236.401     0.9234      100.8
   512    154.822     0.3024      100.8
  1024    175.701     0.1716      100.8
  2048    222.416     0.1086      100.8
  4096    327.020     0.0798      100.8
  8192    562.318     0.0686      100
 16384    960.909     0.0586      157.6
 32768   1998.170     0.0610      311.6
 65536   3732.017     0.0569      620.1
131072   6949.766     0.0530     1237.2


"""

import jax
import jax.numpy as jnp
import mctx
from time import perf_counter

NUM_SIM = 2**6
DISCOUNT = 0.9
ROOT_VALUE = 0
LEFT  = jnp.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
RIGHT = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
VALID = jnp.array([0, 1, 0])
PRIOR = jnp.array([1., 1.])
N_ACTION = 2


def recurrent_fn(params, rng_key, action, embedding):
    _B = embedding.shape[0]
    nxt_left  = embedding @ LEFT.T
    nxt_right = embedding @ RIGHT.T
    new_embedding = jnp.where((action == 0)[:, None], nxt_left, nxt_right)
    reward = jnp.sum(new_embedding * VALID, axis=-1)
    out = mctx.RecurrentFnOutput(
        reward       = reward,
        discount     = jnp.full(_B, DISCOUNT),
        prior_logits = jnp.broadcast_to(PRIOR, (_B, N_ACTION)),
        value        = jnp.full(_B, ROOT_VALUE),
    )
    return out, new_embedding


def run_search(root, key):
    return mctx.muzero_policy(
        params=(), rng_key=key, root=root,
        recurrent_fn=recurrent_fn, num_simulations=NUM_SIM,
    )

# Compile de function (avoid new compilation)
run_jit = jax.jit(run_search)
device = jax.devices()[0]
range_B = [2**i for i in range(18)]
key = jax.random.PRNGKey(0)

print(f"Device: {device} \n")

print(f"{'B':>6} {'time_ms':>10} {'ms/elem':>10} {'peak_MB':>10}")
for B in range_B:
    root = mctx.RootFnOutput(
        prior_logits = jnp.broadcast_to(PRIOR, (B, N_ACTION)),
        value        = jnp.zeros(B),
        embedding    = jnp.broadcast_to(jnp.array([1, 0, 0]), (B, 3)),
    )

    # Warmup: triggers XLA compilation excluded from timing
    run_jit(root, key).action.block_until_ready()
    device.memory_stats()

    t0 = perf_counter()
    out = run_jit(root, key)
    out.action.block_until_ready()   # force async completion
    dt = (perf_counter() - t0) * 1e3

    stats = device.memory_stats()
    peak_mb = stats.get("peak_bytes_in_use", 0) / 1e6

    print(f"{B:>6} {dt:>10.3f} {dt/B:>10.4f} {peak_mb:>10.1f}")





