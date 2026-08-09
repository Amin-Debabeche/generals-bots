"""Policy-value network: a small CNN sized for CPU training throughput, not
inference speed (inference gets reimplemented in pure numpy at export time —
see training/export.py and training/action_space.py for the shared reference
this module's numpy twin must match bit-for-bit-enough).

The network itself only produces raw logits + a value estimate. Legal-action
masking, the pass/build/move index encoding, and observation normalization
all live in training/action_space.py so training code and the exported numpy
agent read from one place instead of two independently-typed copies.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from training.action_space import BUILD_KIND
from training.config import NetworkConfig

# The build-action channel is masked out (illegal) for the entire A/B/C
# curriculum, so it never receives gradient signal and is still sitting at
# its raw random init the moment Stage D flips build_castles_enabled=True --
# where its random-scale logit suddenly competes in the same flat softmax as
# an already-confident, low-entropy movement policy. Starting it strongly
# suppressed (like a backup "pass") means Stage C's movement policy carries
# over intact into Stage D, and building only turns on gradually as real
# reward signal justifies it -- confirmed necessary by two independent runs
# (main4, main5) both regressing hard and non-recovering the moment Stage D
# started, without this init.
_BUILD_LOGIT_INIT_BIAS = -6.0


class PolicyValueNetwork(eqx.Module):
    convs: list[eqx.nn.Conv2d]
    policy_conv: eqx.nn.Conv2d  # 1x1, backbone_out -> num_cell_actions
    pass_linear: eqx.nn.Linear  # backbone_out -> 1
    value_linear1: eqx.nn.Linear
    value_linear2: eqx.nn.Linear

    grid_size: int = eqx.field(static=True)
    num_cell_actions: int = eqx.field(static=True)

    def __init__(self, key: jnp.ndarray, cfg: NetworkConfig = NetworkConfig()):
        channels = (cfg.in_channels,) + cfg.backbone_channels
        num_convs = len(channels) - 1
        keys = jrandom.split(key, num_convs + 4)

        self.convs = [
            eqx.nn.Conv2d(
                channels[i], channels[i + 1], kernel_size=cfg.kernel_size,
                padding=cfg.kernel_size // 2, key=keys[i],
            )
            for i in range(num_convs)
        ]

        backbone_out = channels[-1]
        self.policy_conv = eqx.nn.Conv2d(
            backbone_out, cfg.num_cell_actions, kernel_size=1, key=keys[num_convs]
        )
        self.policy_conv = eqx.tree_at(
            lambda c: c.bias,
            self.policy_conv,
            self.policy_conv.bias.at[BUILD_KIND, 0, 0].set(_BUILD_LOGIT_INIT_BIAS),
        )
        self.pass_linear = eqx.nn.Linear(backbone_out, 1, key=keys[num_convs + 1])
        self.value_linear1 = eqx.nn.Linear(backbone_out, cfg.value_hidden, key=keys[num_convs + 2])
        self.value_linear2 = eqx.nn.Linear(cfg.value_hidden, 1, key=keys[num_convs + 3])

        self.grid_size = cfg.grid_size
        self.num_cell_actions = cfg.num_cell_actions

    def __call__(self, obs_tensor: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """obs_tensor: (in_channels, H, W) float32, already normalized.

        Returns:
            logits: (H*W*num_cell_actions + 1,) flat raw logits, per-cell
                block ordered [row][col][kind] (kind: 0-3 move-all-in dirs,
                4-7 move-half dirs, 8 build), with the single global pass
                logit appended at the end.
            value: () scalar.
        """
        x = obs_tensor
        for conv in self.convs:
            x = jax.nn.relu(conv(x))  # (C, H, W)

        H, W = x.shape[1], x.shape[2]
        cell_logits = self.policy_conv(x)  # (num_cell_actions, H, W)
        cell_logits = jnp.transpose(cell_logits, (1, 2, 0))  # (H, W, num_cell_actions)
        cell_logits_flat = cell_logits.reshape(H * W * self.num_cell_actions)

        pooled = jnp.mean(x, axis=(1, 2))  # (C,) global average pool
        pass_logit = self.pass_linear(pooled)  # (1,)
        logits = jnp.concatenate([cell_logits_flat, pass_logit])

        value_hidden = jax.nn.relu(self.value_linear1(pooled))
        value = self.value_linear2(value_hidden)[0]

        return logits, value
