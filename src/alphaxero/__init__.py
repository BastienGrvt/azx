"""
AlphaZero on mctx + pgx.

Public API: 
    - `Trainer` from a `ConfigTrain` / `ConfigEval`
"""

from .config import Meta, MctsEngine, ConfigTrain, ConfigEval, ConfigAgent
from .trainer import Trainer, TrainerState

__all__ = [
    "Meta", "MctsEngine", "ConfigTrain", "ConfigEval", "ConfigAgent",
    "Trainer", "TrainerState",
]
