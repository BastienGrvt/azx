import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from env import EnvToy
from net import NetToy
from alphazero import make_recurrent_fn, selfplay, train_step

n_batch = 64
n_sim = 256
n_step = 128
n_update = 256
discount = 0.9

key = jax.random.key(0)
net_key, sp_key = jax.random.split(key)

# --- Set up ---
my_env = EnvToy()
num_actions = my_env.num_actions
observation_shape = my_env.observation_shape
network = NetToy(observation_shape, num_actions, key=net_key)
recurrent_fn = make_recurrent_fn(my_env, discount=discount)

# --- Pre-trained game ---
data_game = selfplay(my_env, network, recurrent_fn, sp_key, n_batch, n_sim, n_step, discount=discount)

# --- Overfit that batch ---
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

# --- Network training ---
hist = {"loss": [], "policy": [], "value": []}
for i in range(n_update):
    network, opt_state, m = train_step(network, opt_state, data_game, optimizer)
    hist["loss"].append(float(m["loss"]))
    hist["policy"].append(float(m["policy_loss"]))
    hist["value"].append(float(m["value_loss"]))
    if i % 20 == 0:
        print(f"{i:4d}  loss={m['loss']:.4f}  policy={m['policy_loss']:.4f}  value={m['value_loss']:.4f}")

# --- Post-training game ---
data_game = selfplay(my_env, network, recurrent_fn, sp_key, n_batch, n_sim, n_step, discount=discount)

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



# --- Statistics ---

# Data is already flat [N, *] (no mask, continuous rollout)
obs = data_game["observation"]
val = data_game["value"]

# Group z by discrete state [1,0,0], [0,1,0], [0,0,1] -> 0, 1, 2
states = jnp.argmax(obs, axis=-1)          # 0..10 now
for s in range(11):
    m = states == s
    if m.sum():
        vals = val[m]
        pis = data_game["policy"][m]
        print(f"pos {s:2d}: n={int(m.sum()):3d}  val_mean={vals.mean():.3f}  "f"pi_mean={pis.mean(0)}")


