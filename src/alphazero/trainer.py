"""
Trainer: orchestration class wrapping train / evaluate / save / load.

Owns the mutable training state (network, opt_state) and the single source of
truth for hyperparameters (AZConfig). `recurrent_fn` is DERIVED from
config.discount here -> no double source of truth on discount.

Scope (v1): pure training + evaluation + checkpointing. Periodic eval,
best-model tracking and inference (`act`) are left to the calling script.
"""

import json
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import jax
import equinox as eqx
import optax

from .search import make_recurrent_fn
from .train import train
from .evaluate import evaluate


@dataclass
class AZConfig:
    # --- Rng ----
    seed: int = 42
    # --- optimizer ---
    lr: float = 1e-2
    # --- rollout / training ---
    discount: float = 0.9
    n_iter: int = 5
    n_roll: int = 4
    n_update: int = 8
    n_minibatch: int = 16
    n_batch: int = 4096
    n_step: int = 32
    num_sim: int = 16
    c_value: float = 1.0
    # --- policy (flat kwargs; policy_config refactor deferred) ---
    temperature: float = 1.0
    dirichlet_fraction: float = 0.25
    dirichlet_alpha: float = 0.3


class Trainer:
    """Stateful orchestrator: `fit` mutates (network, opt_state); everything
    else reads the current state."""

    def __init__(self, env, network, config: AZConfig):
        self.env = env
        self.network = network  # built by the caller, injected here
        self.config = config
        self.key = 
        self.optimizer = optax.adam(config.lr)
        self.opt_state = self.optimizer.init(eqx.filter(self.network, eqx.is_array))
        self.recurrent_fn = make_recurrent_fn(env, discount=config.discount)

    # ---- pure training: mutates internal state, returns logs ----
    def fit(self, verbose: bool = True):
        c = self.config
        self.key, train_key = jax.random.split(self.key)
        self.network, self.opt_state, logs = train(
            env=self.env,
            network=self.network,
            recurrent_fn=self.recurrent_fn,
            optimizer=self.optimizer,
            opt_state=self.opt_state,
            key=train_key,
            n_iter=c.n_iter,
            n_roll=c.n_roll,
            n_update=c.n_update,
            n_minibatch=c.n_minibatch,
            n_batch=c.n_batch,
            n_step=c.n_step,
            num_sim=c.num_sim,
            discount=c.discount,
            temperature=c.temperature,
            dirichlet_fraction=c.dirichlet_fraction,
            dirichlet_alpha=c.dirichlet_alpha,
            c_value=c.c_value,
            verbose=verbose,
        )
        return logs

    # ---- greedy evaluation (temperature=0, no exploration noise) ----
    def evaluate(self, key=None, n_batch=None, n_step=None, num_sim=None):
        c = self.config
        if key is None:
            self.key, key = jax.random.split(self.key)
        return evaluate(
            env=self.env,
            network=self.network,
            recurrent_fn=self.recurrent_fn,
            key=key,
            n_batch=n_batch if n_batch is not None else c.n_batch,
            n_step=n_step if n_step is not None else c.n_step,
            num_sim=num_sim if num_sim is not None else c.num_sim,
            discount=1.0,
            temperature=0.0,
            dirichlet_fraction=0.0,
            dirichlet_alpha=c.dirichlet_alpha,
        )

    # ---- checkpointing: dir with model.eqx (net+opt) + config.json ----
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # net + opt_state together -> allows resuming training mid-run
        eqx.tree_serialise_leaves(path / "model.eqx", (self.network, self.opt_state))
        with open(path / "config.json", "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2)

    @classmethod
    def load(cls, path, env, network, key=None):
        # `network` is a template with the right STRUCTURE (built by the caller,
        # same as at __init__); its weights get overwritten by the checkpoint.
        path = Path(path)
        with open(path / "config.json") as f:
            config = AZConfig(**json.load(f))

        self = cls(env, network, config, key if key is not None else jax.random.key(0))
        self.network, self.opt_state = eqx.tree_deserialise_leaves(
            path / "model.eqx", (self.network, self.opt_state)
        )
        return self
