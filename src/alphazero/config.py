from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class Meta:
    env_id: str
    net_id: str
    description: str = ""
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MctsEngine:
    # --- MCTS params ---
    discount: float = 0.9
    num_sim: int = 16
    # --- Policy ---
    kind: str = "muzero"
    # muzero
    temperature: float = 1.0
    dirichlet_fraction: float = 0.25
    dirichlet_alpha: float = 0.3
    pb_c_init: float = 1.25
    pb_c_base: float = 1000.0
    # gumbel
    max_num_considered_actions: int = 16
    gumbel_scale: float = 1.0

    @property
    def policy_kwargs(self):
        if self.kind == "muzero":
            return {
                "temperature": self.temperature,
                "dirichlet_fraction": self.dirichlet_fraction,
                "dirichlet_alpha": self.dirichlet_alpha,
            }
        elif self.kind == "gumbel":
            return {
                "max_num_considered_actions": self.max_num_considered_actions,
                "gumbel_scale": self.gumbel_scale,
            }
        raise ValueError(f"unknown policy kind: {self.kind!r}")


@dataclass
class ConfigTrain:
    seed: int = 42
    mcts: MctsEngine = field(default_factory=MctsEngine)
    optimizer: str = "adam"
    optimizer_kwargs: dict = field(default_factory=lambda: {"learning_rate": 1e-2})
    n_iter: int = 5
    n_roll: int = 4
    n_batch: int = 4096
    n_step: int = 32
    n_update: int = 8
    n_minibatch: int = 16
    c_value: float = 1.0
    
    @property
    def N(self):
        return self.n_roll * self.n_step * self.n_batch

    def __post_init__(self):
        if isinstance(self.mcts, dict):
            self.mcts = MctsEngine(**self.mcts)


@dataclass
class ConfigEval:
    seed: int = 42
    mcts: MctsEngine = field(default_factory=MctsEngine)
    n_batch: int = 4096
    n_step: int = 32
    discount_score: float | None = None

    def __post_init__(self):
        if isinstance(self.mcts, dict):
            self.mcts = MctsEngine(**self.mcts)
        if self.discount_score is None:
            self.discount_score = self.mcts.discount


@dataclass
class ConfigAgent:
    mcts: MctsEngine = field(default_factory=MctsEngine)
    seed: int = 42
    n_step: int = 32

    def __post_init__(self):
        if isinstance(self.mcts, dict):
            self.mcts = MctsEngine(**self.mcts)

