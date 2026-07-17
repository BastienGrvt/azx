import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from rich.console import Console
from rich.table import Table

from env import EnvToy
from net import NetToy
from alphazero import make_recurrent_fn, train_iter, evaluate

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
# Train
temp_train = 1.0
diri_fraction_train = 0.25
diri_alpha_train = 0.3
# Eval game (greedy, no noise)
temp_game = 0.0
diri_fraction_game = 0.0
diri_alpha_game = 0.3

key = jax.random.key(0)
key, train_key = jax.random.split(key)
key, net_key = jax.random.split(key)
key, eval_key = jax.random.split(key)

# --- Set up ---
my_env = EnvToy()
num_actions = my_env.num_actions
observation_shape = my_env.observation_shape
network = NetToy(observation_shape, num_actions, key=net_key)
recurrent_fn = make_recurrent_fn(my_env, discount=discount)

# --- Optimizer ---
optimizer = optax.adam(1e-2)
opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

console = Console()


def run_eval(net, key):
    # greedy rollout for post/pre-training performance
    return evaluate(
        env=my_env, network=net, recurrent_fn=recurrent_fn, key=key,
        n_batch=n_batch, n_step=n_step, num_sim=n_sim,
        discount=1.0,
        temperature=temp_game,
        dirichlet_fraction=diri_fraction_game,
        dirichlet_alpha=diri_alpha_game,
    )


def print_perf(title, metrics):
    t = Table(title=title, header_style="bold cyan", show_edge=False)
    t.add_column("metric", style="dim")
    t.add_column("value", justify="right", style="green")
    t.add_row("games", str(n_batch))
    t.add_row("success_rate", f"{float(metrics['mean_success']):.3f}")
    t.add_row("mean_score",   f"{float(metrics['mean_score']):.3f}")
    t.add_row("mean_length",  f"{float(metrics['mean_length']):.2f}")
    console.print(t)


# --- Pre-training eval (baseline) ---
eval_key, k = jax.random.split(eval_key)
print_perf("Pre-training performance", run_eval(network, k))

# --- Manual training loop over train_iter ---
for idx_iter in range(n_iter):
    train_key, subkey = jax.random.split(train_key)
    network, opt_state, log = train_iter(
        env=my_env,
        network=network,
        recurrent_fn=recurrent_fn,
        optimizer=optimizer,
        opt_state=opt_state,
        key=subkey,
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
    )
    last = log["logs_nn"][-1]
    console.print(
        f"[dim]iter {idx_iter:3d}[/dim]  "
        f"loss=[bold]{last['loss']:.4f}[/bold]  "
        f"policy={last['policy_loss']:.4f}  "
        f"value={last['value_loss']:.4f}  "
        f"val_mean={log['value_mean']:.3f}  "
        f"[cyan]roll {log['time_roll']:4.1f}s[/cyan]  "
        f"[yellow]upd {log['time_updates']:4.1f}s[/yellow]"
    )

# --- Post-training eval ---
eval_key, k = jax.random.split(eval_key)
print_perf("Post-training performance", run_eval(network, k))

# --- Network policy/value probe ---
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
