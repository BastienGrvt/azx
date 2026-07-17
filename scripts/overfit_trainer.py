import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from env import EnvToy
from net import NetToy
from alphazero import Trainer, ConfigTrain, ConfigEval, MctsEngine, Meta

# --- Network and Env ---
def make_network(env, key):
    return NetToy(env.observation_shape, env.num_actions, key=key)
my_env = EnvToy()

# --- Configs
meta = Meta(
    env_id=my_env.id, 
    net_id="NetToy", 
    description="overfit trainer test"
)

# --- Config ---
config_train = ConfigTrain(
    mcts=MctsEngine(
        discount=0.9,
        num_sim=16,
        temperature=1.0,
        dirichlet_fraction=0.25,
        dirichlet_alpha=0.3
    ),
    optimizer="adam",
    optimizer_kwargs={"learning_rate": 2e-2},
    seed=0,
    n_iter=5,
    n_roll=4,
    n_batch=2**12,
    n_step=32,
    n_update=8,
    n_minibatch=16,
    c_value=1.0,
)

config_eval = ConfigEval(
    mcts=MctsEngine(
        discount=1.0,
        num_sim=16,
        temperature=0.0,
        dirichlet_fraction=0.0,
        dirichlet_alpha=0.3
    ),
    seed=0,
    n_batch=2**12,
    n_step=32,
)

# --- Build + run ---
trainer = Trainer(
    meta=meta,
    env=my_env,
    network_fn=make_network,
    config_train=config_train,
    config_eval=config_eval,
    dir_save="runs/overfit_toy",
)
trainer = trainer.init()

t0 = time.perf_counter()
logs = trainer.fit(eval=True, eval_every=1, checkpoint=True, checkpoint_every=1, verbose=True)
print(f"\ndone in {time.perf_counter() - t0:.1f}s — {len(logs)} iterations logged")



