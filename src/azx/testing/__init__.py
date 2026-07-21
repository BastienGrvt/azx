"""
Conformance helpers for user-provided environments.

Convention à la `numpy.testing` / `pandas.testing`: import the submodule and
call the public checker on your own pgx environment before plugging it into the
pipeline.

    from azx.testing import pgx_api_check
    pgx_api_check(MyEnv())
"""

from .pgx_api import pgx_api_check

__all__ = ["pgx_api_check"]
