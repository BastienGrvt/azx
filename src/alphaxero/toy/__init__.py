"""
Toy scaffolding: a minimal pgx environment and Equinox network.

These are development consumables, not part of the core. They let the package
run out-of-the-box (examples, tests) without wiring a real environment or
network.
"""

from .env_toy import EnvToy, State
from .net_toy import NetToy

__all__ = ["EnvToy", "State", "NetToy"]
