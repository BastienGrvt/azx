"""
Navigation sur anneau sous standard pgx (goal-reaching, AlphaZero-like).

Anneau de N positions. But = atteindre le milieu.
  État    : position sur l'anneau (indice scalaire, obs = one-hot [N])
  Actions : 0=gauche, 1=droite, 2=stay   (décalage d'indice mod N, pas de matrice)
  Reward  : +1 en atteignant le milieu (terminal), 0 sinon
  Init    : position aléatoire HORS milieu (sinon partie de longueur 0)
  Fin     : milieu atteint (goal) ; garde-fou timeout à _MAX_STEP

Pas d'ambiguïté temporelle : z(état) = γ^(distance au milieu), déterministe et
lisible dans l'observation -> le floor MSE de l'overfit est nul par construction.
"""

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

_N = 11
_MID = _N // 2
_MAX_STEP = jnp.int32(32)


@dataclass
class State(core.State):
    @property
    def env_id(self):
        return "EnvToy"
    current_player: jax.Array = jnp.int32(0)
    observation: jax.Array = jnp.zeros(_N, dtype=jnp.float32)
    rewards: jax.Array = jnp.float32([0.0])
    terminated: jax.Array = jnp.bool_(False)
    truncated: jax.Array = jnp.bool_(False)
    legal_action_mask: jax.Array = jnp.ones(3, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _pos: jax.Array = jnp.int32(0)     # scalar index on the ring


class EnvToy(core.Env):
    @property
    def id(self):      return "EnvToy"
    @property
    def version(self): return "v0"
    @property
    def num_players(self): return 1

    def _init(self, key: PRNGKey) -> State:
        # random start among the N-1 non-goal positions (avoid length-0 games)
        offset = jax.random.randint(key, (), 1, _N) 
        pos = (_MID + offset) % _N
        return State(_pos=pos)

    def _step(self, state: State, action: Array, key: PRNGKey) -> State:
        # ring move by index shift: left=-1, right=+1, stay=0  (mod N)
        shift = jnp.array([-1, 1, 0])[action]
        new_pos = (state._pos + shift) % _N

        reached = new_pos == _MID
        reward = reached.astype(jnp.float32)
        terminated = reached
        return state.replace(_pos=new_pos, rewards=jnp.array([reward]), terminated=terminated)

    def _observe(self, state: State, player_id: Array) -> Array:
        return jax.nn.one_hot(state._pos, _N)

    def show(self, state: State) -> None:
        print(int(state._pos))



