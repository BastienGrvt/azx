"""
selfplay(env, network, recurrent_fn, key, batch_size, num_simulations, discount=0.99) -> dict_output
dict_output = {
    "observation": obs,     # [T, B, obs_dim]
    "policy": pi,           # [T, B, action_dim]
    "value": val,             # [T, B]
    "mask": mask,           # [T, B]  (True = real step)
    "keys": {
        "action": key_action_t,
        "search": key_search_t,
        },
}

make_recurrent_fn(env, discount: float = 0.99) -> recurrent_fn


train_step(network, opt_state, batch, optimizer, c_value=1.0) -> network, opt_state, {"loss": loss, **loss_dict}
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from env import EnvToy
from net import NetToy
from alphazero import make_recurrent_fn, selfplay, train_step

B = 16
NUM_SIM = 64
N = 256

key = jax.random.key(0)
net_key, sp_key = jax.random.split(key)

# --- Set up ---
my_env = EnvToy()
num_actions = my_env.num_actions
observation_shape = my_env.observation_shape
network = NetToy(observation_shape, num_actions, key=net_key)
recurrent_fn = make_recurrent_fn(my_env, discount=0.99)

# --- Pre-trained game ---
data_game = selfplay(my_env, network, recurrent_fn, sp_key, B, NUM_SIM)

# --- Overfit that batch ---
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

# --- Network training ---
hist = {"loss": [], "policy": [], "value": []}
for i in range(N):
    network, opt_state, m = train_step(network, opt_state, data_game, optimizer)
    hist["loss"].append(float(m["loss"]))
    hist["policy"].append(float(m["policy_loss"]))
    hist["value"].append(float(m["value_loss"]))
    if i % 20 == 0:
        print(f"{i:4d}  loss={m['loss']:.4f}  policy={m['policy_loss']:.4f}  value={m['value_loss']:.4f}")

# --- Post-training game ---
data_game = selfplay(my_env, network, recurrent_fn, sp_key, B, NUM_SIM)

# --- Plot loss vs step ---
plt.figure(figsize=(8, 5))
plt.plot(hist["loss"],   label="total")
plt.plot(hist["policy"], label="policy (CE)")
plt.plot(hist["value"],  label="value (MSE)")
plt.xlabel("training step")
plt.ylabel("loss")
plt.yscale("log")            # overfit -> loss spans orders of magnitude
plt.title("Overfit on a fixed self-play batch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figs/overfit_loss.png", dpi=120)



# --- Which state is focused ---

# Reshape (T, B, *) -> (T*B, *)
obs  = data_game["observation"].reshape(-1, 3)
z    = data_game["value"].reshape(-1)
mask = data_game["mask"].reshape(-1)

# Keep only real samples
obs, z = obs[mask], z[mask]

# Group z by discrete state [1,0,0], [0,1,0], [0,0,1] -> 0, 1, 2
states = jnp.argmax(obs, axis=-1)
for s in [0, 1, 2]:
    zs = z[states == s]
    if zs.size:
        print(f"state {s}: n={zs.size:3d}  z mean={zs.mean():.3f}  std={zs.std():.3f}  "f"min={zs.min():.3f}  max={zs.max():.3f}")

