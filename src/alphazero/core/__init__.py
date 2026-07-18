"""
Core functions
Pure JAX, no device notion, no shard_map, no config object
Every function is callable and testable on a single device
"""

from .search import make_recurrent_fn, make_search_fn
from .rollout import rollout, targets, flatten, selfplay
from .loss import loss_fn, update_step, update_iter
from .metrics import eval_metrics, aggregate_eval

__all__ = [
    "make_recurrent_fn", "make_search_fn",
    "rollout", "targets", "flatten", "selfplay",
    "loss_fn", "update_step", "update_iter",
    "eval_metrics", "aggregate_eval",
]
