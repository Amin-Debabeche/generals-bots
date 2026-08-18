"""Smoke test for training/augment.py against a real (tiny) env -- not a
synthetic-tensor unit test like test_network_transformer.py, since the goal
here is to catch shape/dtype mistakes that only surface once real
Observations (with real fog-of-war, real castle/general placement, a real
pad_to=21 canvas) flow through augment_obs, normalize_augmented, and the
transformer network end to end.

Run directly: python -m training.test_augment
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv
from generals.agents.random_agent import RandomAgent

from training.augment import (
    NUM_BASE_CHANNELS,
    augment_obs,
    init_obs_state,
    normalize_augmented,
    reset_obs_state_where,
    temporal_data,
)
from training.config import TransformerNetworkConfig
from training.network_transformer import TransformerPolicyValueNetwork
from training.rollout import _get_obs_fn


def test_augment_pipeline_end_to_end():
    cfg = TransformerNetworkConfig()
    env = GeneralsEnv(grid_dims=(12, 12), pad_to=cfg.grid_size, truncation=25,
                       perfect_info=True, build_castles=False, deathtouch_turn=None,
                       pool_size=64)
    key = jrandom.PRNGKey(0)
    key, pool_key, state_key = jrandom.split(key, 3)
    pool, _ = env.reset(pool_key)
    state = env.init_state(state_key)

    get_obs = _get_obs_fn(env)
    agent = RandomAgent()
    obs_state0 = init_obs_state(cfg)
    obs_state1 = init_obs_state(cfg)

    net_key, key = jrandom.split(key)
    net = TransformerPolicyValueNetwork(net_key, cfg)

    expected_channels = NUM_BASE_CHANNELS + 2 * cfg.history_size
    num_dones_seen = 0

    for t in range(80):
        key, k0, k1 = jrandom.split(key, 3)
        obs0 = get_obs(state, 0)
        obs1 = get_obs(state, 1)

        augmented0, obs_state0_next = augment_obs(obs0, obs_state0)
        augmented1, obs_state1_next = augment_obs(obs1, obs_state1)
        assert augmented0.shape == (expected_channels, cfg.grid_size, cfg.grid_size)
        assert jnp.all(jnp.isfinite(augmented0)), f"non-finite augmented obs at step {t}"

        normed0 = normalize_augmented(augmented0)
        assert jnp.all(jnp.isfinite(normed0)), f"non-finite normalized obs at step {t}"

        temporal0 = temporal_data(obs_state0_next)
        assert temporal0.shape == (2, cfg.temporal_window)

        logits, value = net(normed0, temporal0)
        expected_logits = cfg.grid_size * cfg.grid_size * cfg.num_cell_actions + 1
        assert logits.shape == (expected_logits,)
        assert jnp.all(jnp.isfinite(logits)), f"non-finite logits at step {t}"
        assert jnp.isfinite(value), f"non-finite value at step {t}"

        action0 = agent.act(obs0, k0)
        action1 = agent.act(obs1, k1)
        actions = jnp.stack([action0, action1])
        timestep, state = env.step(state, actions, pool)
        done = timestep.terminated | timestep.truncated

        obs_state0, obs_state1 = obs_state0_next, obs_state1_next
        if bool(done):
            num_dones_seen += 1
            obs_state0 = init_obs_state(cfg)
            obs_state1 = init_obs_state(cfg)
            state = env.init_state(jrandom.fold_in(state_key, t))

    print(f"ran 80 steps, {num_dones_seen} episode(s) ended, all shapes/finiteness checks passed")
    assert num_dones_seen > 0, "expected at least one episode boundary to exercise the reset path"


def test_reset_obs_state_where():
    cfg = TransformerNetworkConfig()
    single = init_obs_state(cfg)
    batched = jax.tree.map(lambda x: jnp.broadcast_to(x, (3,) + x.shape) + 1.0
                            if x.dtype != jnp.bool_ else jnp.broadcast_to(x, (3,) + x.shape),
                            single)
    # perturb float leaves so we can tell reset apart from a no-op
    batched = batched._replace(
        last_army=jnp.ones((3, cfg.grid_size, cfg.grid_size)),
        opponent_army_history=jnp.ones((3, cfg.temporal_window)),
    )
    dones = jnp.array([True, False, True])
    reset = reset_obs_state_where(batched, dones)
    assert bool(jnp.all(reset.last_army[0] == 0.0))
    assert bool(jnp.all(reset.last_army[1] == 1.0))
    assert bool(jnp.all(reset.last_army[2] == 0.0))
    assert bool(jnp.all(reset.opponent_army_history[1] == 1.0))
    print("reset_obs_state_where: done envs zeroed, others untouched -- OK")


if __name__ == "__main__":
    test_augment_pipeline_end_to_end()
    test_reset_obs_state_where()
