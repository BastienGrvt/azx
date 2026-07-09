"""
selfplay(env, network, recurrent_fn, key, batch_size, num_simulations, discount=0.99) -> dict_output
dict_output = {
    "observation": obs,     # [T, B, obs_dim]
    "policy": pi,           # [T, B, action_dim]
    "value": val,             # [T, B]
    "mask": mask,           # [T, B]  (True = real step)
    "keys": {
        "action": key_action_t,
        "search": key_search_t,
        },
}

make_recurrent_fn(env, discount: float = 0.99) -> recurrent_fn


train_step(network, opt_state, batch, optimizer, c_value=1.0) -> network, opt_state, {"loss": loss, **loss_dict}
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from env import EnvToy
from net import NetToy
from alphazero import selfplay, train_step


# --- Set up the game ---
B = 16
my_env = EnvToy()
network = NetToy(3, 2)

# --- Play the game ---
data_game = selfplay(my_env, network, key, B, NUM_SIM)


# --- Train on the data ---
optimizer = optax.adam(1e-1)
network_filt = eqx.filter(network, eqx.is_array)
opt_state = optimizer.init(network_filt)
for i in range(N):
    network, opt_state, data_loss = train_step(network, opt_state, data_game, optimizer)



