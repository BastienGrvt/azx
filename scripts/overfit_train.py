import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from rich.console import Console
from rich.table import Table

from env import EnvToy
from net import NetToy
from alphazero import make_recurrent_fn, selfplay, train

# --- Hyperparameters ---
# Training
n_iter = 5
n_roll = 4
n_update = 8
n_minibatch = 16
# Self-play
n_batch = 2**12
n_step = 32
n_sim = 16
c_value = 1.0
discount = 0.9

# n_batch, n_step, num_sim, n_roll, n_update, n_minibatch = 8, 4, 4, 2, 2, 2

key = jax.random.key(0)
key, train_key = jax.random.split(key)
key, net_key = jax.random.split(key)
key, play_key = jax.random.split(key)

# --- Set up ---
my_env = EnvToy()
num_actions = my_env.num_actions
observation_shape = my_env.observation_shape
network = NetToy(observation_shape, num_actions, key=net_key)
recurrent_fn = make_recurrent_fn(my_env, discount=discount)

# --- Optimizer ---
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

# --- Full training loop ---
network, opt_state, logs = train(
    my_env, network, recurrent_fn, optimizer, opt_state, train_key,
    n_iter=n_iter, n_roll=n_roll, n_update=n_update, n_minibatch=n_minibatch,
    n_batch=n_batch, n_step=n_step, num_sim=n_sim,
    c_value=c_value, discount=discount, verbose=True,
)

# --- Post-training game ---
data_game = selfplay(
        env=my_env, network=network, recurrent_fn=recurrent_fn, key=play_key,
        n_batch=n_batch, n_step=n_step, num_sim=n_sim,
        discount=discount
    )

# --- Post-Training Statistics ---
console = Console()
stat = Table(title="Value/policy per position", header_style="bold magenta", show_edge=False)
stat.add_column("pos", justify="right", style="dim")
stat.add_column("n", justify="right")
stat.add_column("val_mean", justify="right", style="green")
stat.add_column("pi_mean (L / R / stay)", justify="left")

# Data is already flat [N, *] (no mask, continuous rollout)
obs = data_game["observation"]
val = data_game["value"]

states = jnp.argmax(obs, axis=-1)
for s in range(11):
    m = states == s
    if m.sum():
        vals = val[m]
        pis = data_game["policy"][m].mean(0)
        stat.add_row(
            str(s),
            str(int(m.sum())),
            f"{float(vals.mean()):.3f}",
            "  ".join(f"{float(p):.2f}" for p in pis),
        )

console.print(stat)
