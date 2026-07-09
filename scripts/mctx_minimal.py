"""
Exemple minimal mctx : navigation 1D.

État    : position ∈ {0=gauche, 1=milieu, 2=droite}
Actions : 0=gauche, 1=droite  (clampé aux bords)
Reward  : +1 si nouvelle position == milieu, sinon 0
Prior   : uniforme [1, 1] — pas de réseau
"""

import jax
import jax.numpy as jnp
import numpy as np
import mctx

B       = 10
NUM_SIM = 2**6
DISCOUNT = 0.9
ROOT_VALUE = 0
LEFT  = jnp.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
RIGHT = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
VALID = jnp.array([0, 1, 0])
PRIOR = jnp.array([1., 1.])
N_ACTION = 2

INIT_EMBEDDING = jnp.broadcast_to(jnp.array([1, 0, 0]), (B, 3))
INIT_VALUE = jnp.zeros(B)


def recurrent_fn(params, rng_key, action, embedding):
    _B = embedding.shape[0]

    # Environement
    nxt_left  = embedding @ LEFT.T
    nxt_right = embedding @ RIGHT.T
    new_embedding = jnp.where((action == 0)[:, None], nxt_left, nxt_right)

    # Reward
    reward = jnp.sum(new_embedding * VALID, axis=-1)

    out = mctx.RecurrentFnOutput(
        reward       = reward,
        discount     = jnp.full(_B, DISCOUNT),
        prior_logits = jnp.broadcast_to(PRIOR, (_B, N_ACTION)),
        value        = jnp.full(_B, ROOT_VALUE),
    )
    return out, new_embedding


root = mctx.RootFnOutput(
    prior_logits = jnp.broadcast_to(PRIOR, (B, N_ACTION)),
    value        = INIT_VALUE,
    embedding    = INIT_EMBEDDING,
)

policy_output = mctx.muzero_policy(
    params          = (),
    rng_key         = jax.random.PRNGKey(0),
    root            = root,
    recurrent_fn    = recurrent_fn,
    num_simulations = NUM_SIM,
)

tree = policy_output.search_tree
summary = tree.summary()

visit_probs = summary.visit_probs

print(visit_probs)
