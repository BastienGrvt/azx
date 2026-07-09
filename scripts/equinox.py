import equinox as eqx
import jax
import jax.numpy as jnp

class MLP(eqx.Module):
    layers: list  # list of eqx.nn.Linear -> PyTree leaves

    def __init__(self, in_dim, hidden, out_dim, key):
        # Keys for the 
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(in_dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
            eqx.nn.Linear(hidden, out_dim, key=k3),
        ]

    def __call__(self, x):  # x: (in_dim,) single sample, no batch dim
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

key = jax.random.key(0)
model = MLP(in_dim=4, hidden=16, out_dim=2, key=key)
print(model)

# Single call
x = jnp.ones((4,))
y = model(x)

# Batched call
xs = jnp.ones((32, 4))
ys = jax.vmap(model)(xs)  # batch via vmap -> (32, 2)
