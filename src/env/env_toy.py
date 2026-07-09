"""
Navigation 1D sous standard pgx.

Reprend exactement l'environnement de mctx/mctx_minimal.py et le réécrit
en suivant le pattern State/Env de pgx.

État    : position one-hot [3]  — [1,0,0]=gauche, [0,1,0]=milieu, [0,0,1]=droite
Actions : 0=gauche, 1=droite   (matrices de transition vectorisées)
Reward  : +1 si nouvelle position == milieu, sinon 0
"""

import jax
import jax.numpy as jnp

import pgx
import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

_LEFT  = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=jnp.float32)
_RIGHT = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
_MASK = jnp.array([[0, 1, 1], [1, 1, 0]], dtype=jnp.float32)
_VALID = jnp.array([0, 1, 0], dtype=jnp.float32)
_MAX_STEP = jnp.int32(8)


@dataclass
class State(core.State):
    @property
    def env_id(self):
        return "MyEnv"
    current_player: jax.Array = jnp.int32(0)
    observation: jax.Array = jnp.array([1, 0, 0], dtype=jnp.float32)
    rewards: jax.Array = jnp.float32([0.0])
    terminated: jax.Array = jnp.bool_(False)
    truncated: jax.Array = jnp.bool_(False)
    legal_action_mask: jax.Array = jnp.array([True, True], dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _pos: jax.Array = jnp.array([1, 0, 0], dtype=jnp.float32)


class EnvToy(core.Env):
    @property
    def id(self):      return "EnvTest"
    @property
    def version(self): return "v0"
    @property
    def num_players(self): return 1

    def _init(self, key: PRNGKey) -> State:
        return State()

    def _step(self, state: State, action: Array, key: PRNGKey) -> State:
        nxt_left  = state._pos @ _LEFT.T
        nxt_right = state._pos @ _RIGHT.T
        new_pos = jnp.where(action == 0, nxt_left, nxt_right)
        reward = jnp.dot(new_pos, _VALID)
        legal_action_mask = (new_pos @ _MASK.T).astype(jnp.bool_)
        terminated = state._step_count > _MAX_STEP - 1
        return state.replace(_pos=new_pos, legal_action_mask=legal_action_mask, rewards=jnp.array([reward]), terminated=terminated)

    def _observe(self, state: State, player_id: Array) -> Array:
        return state._pos  # pgx core fait state.replace(observation=...) lui-même

    def show(self, state: State) -> None:
        print(jnp.argmax(state._pos, axis=-1))

