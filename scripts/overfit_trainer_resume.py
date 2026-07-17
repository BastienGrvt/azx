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

trainer_resumed = Trainer.resume(
    dir_save="runs/overfit_toy",
    env=my_env,
    network_fn=make_network,
    label="3",
)

t0 = time.perf_counter()
logs_resumed = trainer_resumed.fit( eval=True, eval_every=1, checkpoint=True, checkpoint_every=1, verbose=True)
print(f"\nresume done in {time.perf_counter() - t0:.1f}s — " f"{len(logs_resumed)} total iterations logged")
