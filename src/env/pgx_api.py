import jax
import jax.numpy as jnp


def pgx_api_check(env, num=100, batch_size=4):
    """Validate a custom pgx env: dtypes, contract, single + batched rollout."""
    _test_single(env, num)
    _test_batch(env, num, batch_size)
    # print("OK")


def _validate_state(state):
    # dtype contract
    assert state.current_player.dtype == jnp.int32
    assert state.terminated.dtype == jnp.bool_
    assert state.rewards.dtype == jnp.float32
    assert state.legal_action_mask.dtype == jnp.bool_
    # no unexpected public attributes
    public = {"current_player", "observation", "rewards", "terminated", "truncated", "legal_action_mask"}
    for k in state.__dict__:
        if not k.startswith("_"):
            assert k in public, f"unexpected public field: {k}"
    # all-True mask at terminal (avoids div-by-zero on prob normalization)
    if state.terminated:
        assert state.legal_action_mask.all()


def _random_action(key, mask):
    # sample uniformly among legal actions
    logits = jnp.where(mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, logits)


def _test_single(env, num):
    init, step = jax.jit(env.init), jax.jit(env.step)
    rng = jax.random.PRNGKey(0)

    for _ in range(num):
        rng, k = jax.random.split(rng)
        state = init(k)
        # init invariants
        assert (state.rewards == 0).all(), "reward must be zero at init"
        assert state.legal_action_mask.sum() != 0, "empty mask at init"
        assert state._step_count == 0
        _validate_state(state)

        steps = 0
        while not state.terminated:
            rng, ka, ks = jax.random.split(rng, 3)
            action = _random_action(ka, state.legal_action_mask)
            state = step(state, action, ks)
            steps += 1
            assert state._step_count == steps
            _validate_state(state)
            if steps > 10_000:  # safety for non-terminating envs
                break


def _test_batch(env, num, batch_size):
    init = jax.jit(jax.vmap(env.init))
    step = jax.jit(jax.vmap(env.step))
    rng = jax.random.PRNGKey(1)

    for _ in range(num):
        rng, k = jax.random.split(rng)
        state = init(jax.random.split(k, batch_size))
        n = 0
        while not state.terminated.all():
            rng, ka, ks = jax.random.split(rng, 3)
            action = jax.vmap(_random_action)(jax.random.split(ka, batch_size), state.legal_action_mask)
            state = step(state, action, jax.random.split(ks, batch_size))
            n += 1
            if n > 10_000:
                break
