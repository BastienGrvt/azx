"""
Test environements pgx api
"""

import jax
import jax.numpy as jnp

from alphaxero.testing import pgx_api_check


def test_pgx_api(env):
    pgx_api_check(env)

