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
    discount: float = 0.9
    num_sim: int = 16
    temperature: float = 1.0
    dirichlet_fraction: float = 0.25
    dirichlet_alpha: float = 0.3


@dataclass
class ConfigTrain:
    mcts: MctsEngine = field(default_factory=MctsEngine)
    optimizer: str = "adam"
    optimizer_kwargs: dict = field(default_factory=lambda: {"learning_rate": 1e-2})
    seed: int = 42
    n_iter: int = 5
    n_roll: int = 4
    n_batch: int = 4096
    n_step: int = 32
    n_update: int = 8
    n_minibatch: int = 16
    c_value: float = 1.0

    def __post_init__(self):
        if isinstance(self.mcts, dict):
            self.mcts = MctsEngine(**self.mcts)


@dataclass
class ConfigEval:
    mcts: MctsEngine = field(default_factory=MctsEngine)
    seed: int = 42
    n_batch: int = 4096
    n_step: int = 32

    def __post_init__(self):
        if isinstance(self.mcts, dict):
            self.mcts = MctsEngine(**self.mcts)


@dataclass
class ConfigAgent:
    mcts: MctsEngine = field(default_factory=MctsEngine)
    seed: int = 42
    n_step: int = 32

    def __post_init__(self):
        if isinstance(self.mcts, dict):
            self.mcts = MctsEngine(**self.mcts)

