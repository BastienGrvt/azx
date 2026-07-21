"""
Minimal end-to-end AlphaZero training on the toy environment.

High-level pipeline:
    1. Pick an environment (pgx) and a network factory (Equinox).
    2. Describe the run: `Meta` (bookkeeping) + `ConfigTrain` / `ConfigEval`.
    3. Build a `Trainer`, `.init()` fresh state, then `.fit()`.

`fit` runs `n_iter` iterations of: self-play collection (MCTS via mctx) ->
network update (policy + value loss) -> periodic greedy evaluation. Metadata,
configs, logs, checkpoints and the final network are written under `dir_save`.

`EnvToy` is a goal-reaching ring env designed to overfit quickly, so this
example runs on CPU in seconds and doubles as a smoke test of the pipeline.

Run:  shell-run cuda-wsl uv run python examples/training.py
"""

import time

from alphaxero import Trainer, ConfigTrain, ConfigEval, MctsEngine, Meta
from alphaxero.toy import EnvToy, NetToy


# --- Environment + network factory ---
# The trainer owns network creation, it calls `network_fn(env, key)` internally.
env = EnvToy()


def make_network(env, key):
    return NetToy(env.observation_shape, env.num_actions, key=key)


# --- Run metadata (bookkeeping only) ---
meta = Meta(
    env_id=env.id,
    net_id="NetToy",
    description="toy AlphaZero training example",
)

# --- Training config: exploratory MCTS (Dirichlet noise, temperature) ---
config_train = ConfigTrain(
    mcts=MctsEngine(
        discount=0.9,
        num_sim=16,
        kind="muzero",
        temperature=1.0,
        dirichlet_fraction=0.25,
        dirichlet_alpha=0.3,
        pb_c_init=1.25,
        pb_c_base=800,
    ),
    optimizer="adam",
    optimizer_kwargs={"learning_rate": 2e-2},
    seed=0,
    n_iter=3,        # training iterations
    n_roll=4,        # self-play rollouts per iteration
    n_batch=2**12,   # games played in parallel
    n_step=32,       # max steps per game
    n_update=8,      # gradient passes over collected data
    n_minibatch=16,
    c_value=1.0,     # value-loss weight
)

# --- Eval config: greedy MCTS (no noise, zero temperature) ---
config_eval = ConfigEval(
    mcts=MctsEngine(
        discount=1.0,
        num_sim=16,
        kind="muzero",
        temperature=0.0,
        dirichlet_fraction=0.0,
    ),
    seed=0,
    n_batch=2**12,
    n_step=32,
    discount_score=1.0,
)

# --- Build, initialize, and fit ---
trainer = Trainer(
    meta=meta,
    env=env,
    network_fn=make_network,
    config_train=config_train,
    config_eval=config_eval,
    dir_save="local/trainings/toy",
)
trainer = trainer.init()

t0 = time.perf_counter()
logs = trainer.fit(eval_every=1, checkpoint_every=1, verbose=True)
print(f"\ndone in {time.perf_counter() - t0:.1f}s \t {len(logs)} iterations logged")
