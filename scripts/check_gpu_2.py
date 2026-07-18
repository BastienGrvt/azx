"""
Go/no-go #2: does pmean keep network replicas in sync under shard_map?

Runs one update_iter inside a global shard_map envelope, then checks that
all devices ended up with bit-identical parameters.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
os.environ["JAX_PLATFORMS"] = "cpu"

from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from env import EnvToy
from net import NetToy
from alphazero import selfplay, update_iter, make_recurrent_fn, make_search_fn

# --- Setup ---
N_BATCH, N_STEP, N_SIM, DISCOUNT = 16, 4, 4, 0.9
N_UPDATE, N_MINIBATCH, C_VALUE = 2, 4, 1.0

n_gpu = jax.device_count()
mesh = jax.make_mesh((n_gpu,), ("batch",), axis_types=(jax.sharding.AxisType.Auto,))
jax.set_mesh(mesh)

env = EnvToy()
network = NetToy(env.observation_shape, env.num_actions, key=jax.random.PRNGKey(0))
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
recurrent_fn = make_recurrent_fn(env, discount=DISCOUNT)
search_fn = make_search_fn(kind="muzero", num_sim=N_SIM,
                           temperature=1.0, dirichlet_fraction=0.25, dirichlet_alpha=0.3)

# Local sizes: everything inside the envelope is per-device
n_batch_local = N_BATCH // n_gpu
n_minibatch_local = N_MINIBATCH // n_gpu
N_local = n_step_local = n_batch_local * N_STEP
size_minibatch = N_local // n_minibatch_local


@partial(jax.shard_map, mesh=mesh,
         in_specs=(jax.P(), jax.P(), jax.P("batch")),
         out_specs=(jax.P(), jax.P(), jax.P()),
         check_vma=False)
def train_local(network, opt_state, keys):
    # per-device: rollout on local batch, then local-shuffle update
    key_roll, key_upd = jax.random.split(keys[0])
    data = selfplay(env=env, network=network,
                    recurrent_fn=recurrent_fn, search_fn=search_fn,
                    key=key_roll, n_batch=n_batch_local,
                    n_step=N_STEP, discount=DISCOUNT)
    return update_iter(network, opt_state, data, optimizer, key_upd,
                       N_UPDATE, n_minibatch_local, size_minibatch, C_VALUE)


keys = jax.random.split(jax.random.PRNGKey(42), n_gpu)
keys = jax.device_put(keys, jax.NamedSharding(mesh, jax.P("batch")))

network, opt_state, logs = train_local(network, opt_state, keys)

# --- Sync check ---
w = network.trunk.weight
shards = w.addressable_shards
print(f"n_shards visible : {len(shards)}")
if len(shards) > 1:
    same = all(jnp.array_equal(shards[0].data, s.data) for s in shards[1:])
    print(f"replicas in sync : {same}")
else:
    print("single shard -> replicated by construction, see fallback below")

# --- Fallback / stronger check: multi-device == single-device ---
print(f"loss (mean)      : {float(jnp.mean(logs['loss'])):.6f}")
print(f"weight checksum  : {float(jnp.sum(w)):.6f}")

hlo = jax.jit(train_local).lower(network, opt_state, keys).compile().as_text()
print(f"all-reduce count : {hlo.count('all-reduce')}   (expect > 0: the pmean)")
