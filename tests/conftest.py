import jax
import pytest

from azx.toy import EnvToy, NetToy


# ------ Object to test -----

envs = {
        "params": [EnvToy],
        "ids": ["EnvToy"]
    }

networks = {
        "params": [NetToy],
        "ids": ["NetToy"]
    }


# ------ Fixed object -----

@pytest.fixture
def key():
    return jax.random.key(0)

@pytest.fixture
def env_toy():
    return EnvToy()

@pytest.fixture
def net_toy(env_toy, key):
    num_actions = env_toy.num_actions
    observation_shape = env_toy.observation_shape
    return NetToy(observation_shape, num_actions, key=key)

# ----- All Env and Networks ------

@pytest.fixture(**envs)
def env(request):
    the_env = request.param
    return the_env()

@pytest.fixture(**networks)
def network(request, env_toy, key):
    num_actions = env_toy.num_actions
    observation_shape = env_toy.observation_shape
    the_network = request.param
    return the_network(observation_shape, num_actions, key=key)

