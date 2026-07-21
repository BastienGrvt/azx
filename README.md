# AlphaXero

A compact AlphaZero trainer in JAX, built on [`mctx`](https://github.com/google-deepmind/mctx) for the MCTS search and the [`pgx`](https://github.com/sotetsuk/pgx) environment standard.

> **Status: work in progress.** 
> APIs are unstable and will change without notice. 
> Personal research/learning codebase not a maintained library.

## Installation

Requires Python ≥ 3.14. 
Dependency management uses [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/BastienGrvt/alphaxero
cd alphaxero
uv sync
```
`jax[cuda13]` is used for GPU, toy example below also runs on CPU.

## Usage

### Toy environment and network example

Run `uv run python examples/traiing.py`.

### Customized environment

The package ships a toy environment (`EnvToy`) and network (`NetToy`) running out-of-the-box. 
See [`examples/training.py`](examples/training.py) for a commented end-to-end run:

```python
from alphaxero import Trainer, ConfigTrain, ConfigEval, Meta
from alphaxero.toy import EnvToy, NetToy

env = EnvToy()
make_network = lambda env, key: NetToy(env.observation_shape, env.num_actions, key=key)

trainer = Trainer(
    meta=Meta(env_id=env.id, net_id="NetToy"),
    env=env,
    network_fn=make_network,
    config_train=ConfigTrain(n_iter=3),
    config_eval=ConfigEval(),
    dir_save="local/trainings/toy",
).init()

trainer.fit()
```

```bash
uv run python examples/training.py
```

To plug in your own `pgx` environment, validate it against the API contract first:

```python
from alphaxero.testing import pgx_api_check
pgx_api_check(MyEnv())
```

## Repository structure

```
src/alphaxero/
├── core/       # pure, mesh-agnostic JAX functions (search, rollout, loss, metrics)
├── parallel/   # sharding and multi-device orchestration
├── toy/        # EnvToy / NetToy — toy env and network (dev scaffolding)
├── testing/    # pgx API conformance check for user environments
├── config.py   # run configuration (Meta, MctsEngine, ConfigTrain/Eval/Agent)
└── trainer.py  # Trainer: train / evaluate / save / load orchestration
examples/       # pedagogical training scripts
tests/          # minimal tests
```

## Stack

JAX · Equinox · mctx · pgx · Optax · Rich.

## License

[Apache 2.0](LICENSE).


