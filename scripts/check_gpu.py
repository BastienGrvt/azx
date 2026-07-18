"""
Go/no-go: can mctx + pgx run inside a shard_map (Manual sharding mode)?

Wraps selfplay in a global shard_map. Each device sees only its own slice.
If this compiles and runs, the global-envelope architecture is viable.
"""

# Force 4 fake CPU devices when no multi-GPU is available.
# Comment out the two lines below to test on real GPUs.
import os
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["JAX_PLATFORMS"] = "cpu"

from functools import partial
import jax
import jax.numpy as jnp

from env import EnvToy
from net import NetToy
from alphazero import selfplay, make_recurrent_fn, make_search_fn


# --- Setup ---
N_BATCH, N_STEP, N_SIM, DISCOUNT = 64, 8, 8, 0.9

n_gpu = jax.device_count()
print(f"devices: {n_gpu} x {jax.devices()[0].device_kind}")
assert N_BATCH % n_gpu == 0

mesh = jax.make_mesh((n_gpu,), ("batch",), axis_types=(jax.sharding.AxisType.Auto,))
jax.set_mesh(mesh)

env = EnvToy()
network = NetToy(env.observation_shape, env.num_actions, key=jax.random.PRNGKey(0))
recurrent_fn = make_recurrent_fn(env, discount=DISCOUNT)
search_fn = make_search_fn(kind="muzero", num_sim=N_SIM,
                           temperature=1.0, dirichlet_fraction=0.25, dirichlet_alpha=0.3)


# --- The envelope: per-device selfplay, local batch only ---
@partial(
    jax.shard_map, mesh=mesh,
    in_specs=(jax.P(), jax.P("batch")),
    out_specs=jax.P("batch"),
    check_vma=False,
)
def selfplay_sharded(network, keys):
    # keys: [1] on each device -> local rollout of N_BATCH // n_gpu games
    data = selfplay(
        env=env, network=network,
        recurrent_fn=recurrent_fn, search_fn=search_fn,
        key=keys[0],
        n_batch=N_BATCH // n_gpu,
        n_step=N_STEP,
        discount=DISCOUNT,
    )
    return data["value"]


# --- Run ---
keys = jax.random.split(jax.random.PRNGKey(42), n_gpu)
keys = jax.device_put(keys, jax.NamedSharding(mesh, jax.P("batch")))

value = selfplay_sharded(network, keys)

print(f"\nOK — shape {value.shape}, mean {value.mean():.4f}")
print(f"sharding: {value.sharding.spec}")
jax.debug.visualize_array_sharding(value.reshape(n_gpu, -1))

# --- Collectives check: should be empty for a pure rollout ---
hlo = jax.jit(selfplay_sharded).lower(network, keys).compile().as_text()
for op in ("all-gather", "all-to-all", "all-reduce", "collective-permute"):
    print(f"{op:20s} {hlo.count(op)}")
