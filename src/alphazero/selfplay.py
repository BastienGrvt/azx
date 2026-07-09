"""
Self-play: jittable rollout via lax.scan on a fixed horizon, then flatten + drop padding.

Per step (scan body):
  1. root eval: network(state) -> (prior, value), mask illegal actions
  2. mctx search -> improved policy (visit distribution)
  3. collect target BEFORE step: (obs, policy, mask=~done)
  4. play sampled action on the REAL env
  5. reward AFTER step (zeroed by ~done), advance cumulative done

Value target = discounted return z_t = Σ γ^k r_{t+k}, computed via backward scan.
After rollout, transitions are flattened [T,B]->[N] and padded steps (mask=False)
are dropped. `flat[mask]` is data-dependent, so it runs OUTSIDE the jitted core.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import mctx


@eqx.filter_jit
def _rollout(env, network, recurrent_fn, key, batch_size, num_simulations, horizon, discount):
    init = jax.vmap(env.init)
    step = jax.vmap(env.step)

    key, sub = jax.random.split(key)
    keys = jax.tandom.split(sub, batch_size)
    state = init(keys)
    done = jnp.zeros(batch_size, dtype=jnp.bool_)

    def body(carry, _):
        state, done, key = carry
        key, search_key, action_key = jax.random.split(key, 3)

        # Root
        prior_logits, value = eqx.filter_vmap(network)(state.observation)
        prior_logits = jnp.where(state.legal_action_mask, prior_logits, -jnp.inf)
        root = mctx.RootFnOutput(
            prior_logits=prior_logits,
            value=value,
            embedding=state
        )

        # MCTS
        policy_output = mctx.muzero_policy(
            params=network,
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
        )

        # Pre-collection data
        obs_out, pi_out, mask_out = state.observation, policy_output.action_weights, ~done

        # Play the move
        state = step(state, policy_output.action, jax.random.split(action_key, batch_size))

        # Post-collection data
        rew_out = state.rewards[:, 0] * (~done)
        done = done | state.terminated
        return (state, done, key), (obs_out, pi_out, rew_out, mask_out)
    
    # final_carry, ys = lax.scan(body, init_carry, xs, length=N)
    _, (obs, pi, rew, mask) = jax.lax.scan(body, (state, done, key), xs=None, length=horizon)

    # backward discounted return
    def scan_return(carry, r):
        carry = r + discount * carry
        return carry, carry
    _, val_rev = jax.lax.scan(scan_return, jnp.zeros(batch_size), rew[::-1])
    val = val_rev[::-1]

    return {"observation": obs, "policy": pi, "value": val, "mask": mask}


def _flatten_mask(data):
    # [T, B, *] -> [T*B, *] then drop padded steps (mask=False). Not jittable: shape is data-dependent.
    T, B = data["mask"].shape
    m = data["mask"].reshape(T * B)
    flat = {
        "observation": data["observation"].reshape(T * B, *data["observation"].shape[2:]),
        "policy":      data["policy"].reshape(T * B, data["policy"].shape[-1]),
        "value":       data["value"].reshape(T * B),
    }
    return {k: v[m] for k, v in flat.items()}


def selfplay(env, network, recurrent_fn, key, batch_size, num_simulations, horizon, discount=0.99):
    data = _rollout(env, network, recurrent_fn, key, batch_size, num_simulations, horizon, discount)
    return _flatten_mask(data)



