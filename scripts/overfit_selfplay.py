import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from env import EnvToy
from net import NetToy
from alphazero import make_recurrent_fn, selfplay, update_step

# --- Hyperparameters ---
discount = 0.9
n_batch = 2**14
n_step = 32
n_sim = 16
n_update = 128
c_value = 1.0
temperature = 1.0
diri_fraction = 0.25
diri_alpha = 0.3

key = jax.random.key(0)
net_key, sp_key = jax.random.split(key)

# --- Set up ---
my_env = EnvToy()
num_actions = my_env.num_actions
observation_shape = my_env.observation_shape
network = NetToy(observation_shape, num_actions, key=net_key)
recurrent_fn = make_recurrent_fn(my_env, discount=discount)

# --- Collecting data ---
data_game = selfplay(
    env=my_env, network=network, recurrent_fn=recurrent_fn, key=sp_key,
    n_batch=n_batch, n_step=n_step, num_sim=n_sim,
    discount=discount,
    temperature=temperature,
    dirichlet_fraction=diri_fraction,
    dirichlet_alpha=diri_alpha,
)

# --- Overfit that batch ---
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

# --- Network training ---
hist = {"loss": [], "policy": [], "value": []}
for i in range(n_update):
    network, opt_state, m = update_step(
        network=network,
        opt_state=opt_state,
        data_minibatch=data_game,
        optimizer=optimizer,
        c_value=c_value,
    )
    hist["loss"].append(float(m["loss"]))
    hist["policy"].append(float(m["policy_loss"]))
    hist["value"].append(float(m["value_loss"]))
    if i % 20 == 0:
        print(f"{i:4d}  loss={m['loss']:.4f}  policy={m['policy_loss']:.4f}  value={m['value_loss']:.4f}")

# --- Post-training game (reuses sp_key on purpose: same states, check fit) ---
data_game = selfplay(
    env=my_env, network=network, recurrent_fn=recurrent_fn, key=sp_key,
    n_batch=n_batch, n_step=n_step, num_sim=n_sim,
    discount=discount,
    temperature=temperature,
    dirichlet_fraction=diri_fraction,
    dirichlet_alpha=diri_alpha,
)


# --- Network Statistics ---
N = observation_shape[0]
all_obs = jnp.eye(N)                                   # one-hot per ring position
logits, values = eqx.filter_vmap(network)(all_obs)
policies = jax.nn.softmax(logits, axis=-1)

for s in range(N):
    pi = policies[s]
    print(f"pos {s:2d}: val={float(values[s]):.3f}  "
          f"pi(L/R/stay)=[{float(pi[0]):.2f} {float(pi[1]):.2f} {float(pi[2]):.2f}]")
