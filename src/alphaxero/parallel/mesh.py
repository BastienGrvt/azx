"""
Device mesh and derived local sizes.

Config sizes are GLOBAL, per-device sizes are derived here and nowhere else.
Nothing in `core/` knows about the mesh: this is the only place where
global -> local translation happens.
"""

import jax
import equinox as eqx
from rich.console import Console


class ShardingPlan:
    """
    Holds the mesh and every size derived from it
    """

    def __init__(self, config_train, config_eval, mesh=None):
        self.config_train = config_train
        self.config_eval = config_eval
        ct, ce = config_train, config_eval

        # --- Mesh ---
        if mesh is None:
            mesh = jax.make_mesh(
                (jax.device_count(),), ("batch",),
                axis_types=(jax.sharding.AxisType.Auto,),
            )
        self.mesh = mesh
        self.axis = mesh.axis_names[0]
        self.n_gpu = mesh.shape[self.axis]

        # --- Validate divisibility ---
        self._validate()

        # --- Get local size ---
        self.n_batch_local_train = ct.n_batch // self.n_gpu
        self.n_batch_local_eval = ce.n_batch // self.n_gpu
        self.n_minibatch_local = ct.n_minibatch // self.n_gpu
        self.N_local = ct.n_roll * ct.n_step * self.n_batch_local_train
        self.size_minibatch_local = self.N_local // self.n_minibatch_local

    # --- Specs and shardings---
    # Set the dispatch: shard_map -> `jax.P`
    # Put the dispatch: device_put -> `NamedSharding`

    @property
    def P_rep(self):
        return jax.P()

    @property
    def P_shard(self):
        return jax.P(self.axis)

    @property
    def replicated(self):
        return jax.NamedSharding(self.mesh, self.P_rep)

    @property
    def sharded(self):
        return jax.NamedSharding(self.mesh, self.P_shard)

    # --- Helpers ---

    # Split and put one key per device
    def split_keys(self, key):
        return jax.device_put(jax.random.split(key, self.n_gpu), self.sharded)

    # Copy the pytree over the devices
    def replicate(self, pytree):
        return eqx.filter_shard(pytree, self.replicated)

    def _validate(self):
        """
        Validate the `n_*` compatibilities:
            - Hard errors: sharding requires exact divisibility (shapes must be static)
            - Warning: leftover samples silently dropped from every update (minibatch size)
        """
        n_gpu = self.n_gpu
        ct, ce = self.config_train, self.config_eval
        problems = []

        # --- Hard: batch and minibatch counts must split across devices ---
        if ct.n_batch % n_gpu:
            problems.append(f"train n_batch={ct.n_batch} % n_gpu={n_gpu}")
        if ce.n_batch % n_gpu:
            problems.append(f"eval n_batch={ce.n_batch} % n_gpu={n_gpu}")
        if ct.n_minibatch % n_gpu:
            problems.append(f"n_minibatch={ct.n_minibatch} % n_gpu={n_gpu}")
        if problems:
            raise ValueError("sharding divisibility violated: " + "; ".join(problems))

        # --- Soft: leftover samples dropped by the minibatch split ---
        N_local = ct.n_roll * ct.n_step * (ct.n_batch // n_gpu)
        n_minibatch_local = ct.n_minibatch // n_gpu
        rem = N_local % n_minibatch_local
        if rem:
            size_minibatch = N_local // n_minibatch_local
            Console().print(
                f"[yellow]⚠ n_minibatch_local={n_minibatch_local} does not divide "
                f"N_local={N_local} (n_roll·n_step·n_batch/n_gpu).[/yellow]  "
                f"size_minibatch={size_minibatch}, "
                f"[bold]{rem}[/bold] samples dropped per update per device "
                f"([dim]{100 * rem / N_local:.2f}% of data[/dim])."
            )



