import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from rich.console import Console
from rich.table import Table

from env import EnvToy
from net import NetToy
from alphazero import make_recurrent_fn, train

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
# Train policy
temp_train = 1.0
diri_fraction_train = 0.25
diri_alpha_train = 0.3
# Eval
eval = True
eval_every = 1
eval_n_batch = 2**12
eval_n_step = 16
eval_num_sim = 16

key = jax.random.key(0)
key, train_key = jax.random.split(key)
key, net_key = jax.random.split(key)

# --- Set up ---
my_env = EnvToy()
num_actions = my_env.num_actions
observation_shape = my_env.observation_shape
network = NetToy(observation_shape, num_actions, key=net_key)
recurrent_fn = make_recurrent_fn(my_env, discount=discount)

# --- Optimizer ---
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

# --- Full training loop (eval woven in) ---
network, opt_state, logs = train(
    env=my_env,
    network=network,
    recurrent_fn=recurrent_fn,
    optimizer=optimizer,
    opt_state=opt_state,
    key=train_key,
    n_iter=n_iter,
    n_roll=n_roll,
    n_update=n_update,
    n_minibatch=n_minibatch,
    n_batch=n_batch,
    n_step=n_step,
    num_sim=n_sim,
    discount=discount,
    c_value=c_value,
    temperature=temp_train,
    dirichlet_fraction=diri_fraction_train,
    dirichlet_alpha=diri_alpha_train,
    eval=eval,
    eval_every=eval_every,
    eval_n_batch=eval_n_batch,
    eval_n_step=eval_n_step,
    eval_num_sim=eval_num_sim,
    verbose=True,
)

console = Console()

# --- Network policy/value ---
N = observation_shape[0]
all_obs = jnp.eye(N)
logits, values = eqx.filter_vmap(network)(all_obs)
policies = jax.nn.softmax(logits, axis=-1)

probe = Table(title="Network value/policy per position", header_style="bold magenta", show_edge=False)
probe.add_column("pos", justify="right", style="dim")
probe.add_column("value", justify="right", style="green")
probe.add_column("policy (L / R / stay)", justify="left")

for s in range(N):
    probe.add_row(
        str(s),
        f"{float(values[s]):.3f}",
        "  ".join(f"{float(p):.2f}" for p in policies[s]),
    )
console.print(probe)
