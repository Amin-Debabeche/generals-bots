"""Transformer policy-value network, adapted from strakam/AverageJoe's
HistoryTransformer (networks/transformer.py in their repo -- the #1-ranked
real generals.io ladder bot, same author as our base engine) for this
project's action-space convention and castle-building ruleset. See
~/.claude/plans/drifting-questing-babbage.md for the full design rationale.

Adaptations from AverageJoe's original, each load-bearing:

1. CELL-MAJOR policy output, not their kind-major. AverageJoe's own
   unpatchify (`action_logits.transpose(2,0,3,1,4).reshape(9,p,p)`) produces
   a (kind, row, col) tensor -- correct for THEIR encode_action/decode_action
   (networks/common.py), which is unrelated to ours. This project's
   training/action_space.py's decode_action_index assumes a flat index
   `row*(W*9) + col*9 + kind`, i.e. a (H, W, 9) array flattened row-major
   (kind fastest-varying). The transpose here
   (`.transpose(0, 3, 1, 4, 2)`, see _unpatchify_policy) produces exactly
   that layout. Getting this transpose wrong would still train (loss drops,
   nothing crashes) but every downstream index would silently decode to the
   wrong (row, col, kind) -- verified correct via
   test_network_transformer.py's decode-roundtrip test, which must pass
   before this module is used for anything else.

2. A separate global pass logit (from the value/CLS token), not AverageJoe's
   per-cell pseudo-pass (their kind==8 at every cell). Matches
   training/action_space.py's num_logits = H*W*9 + 1 convention that the
   rest of this pipeline (rollout.py, ppo.py, magnet.py, export.py,
   competition/agents/my_bot) is already built on.

3. BUILD channel (kind=8, training/action_space.py's BUILD_KIND) gets the
   same strongly-negative bias init training/network.py's CNN needed
   (_BUILD_LOGIT_INIT_BIAS, confirmed necessary by two independent runs):
   BUILD is masked illegal through curriculum stages A-C, so it never gets
   gradient and is still at raw init when Stage D's mask flips it on,
   corrupting argmax from iteration 1 if left unaddressed.

4. patch_size defaults to 3, not AverageJoe's 1 (for our 21x21 board) --
   per-cell action resolution comes from the per-patch head's 9*M*M output +
   unpatchify, not from patch_size itself (that parameter only controls
   transformer sequence length; attention is O(n^2), so patch_size=1 costs
   ~9x the attention FLOPs of patch_size=3 for zero action-space benefit).

5. Plain scalar value head (MSE), not AverageJoe's optional HL-Gauss
   categorical head -- deferred per the plan (their own production config
   uses MSE too; HL-Gauss isn't actually load-bearing for their ladder rank).

Takes a pre-normalized, pre-augmented (n_channels, grid_size, grid_size)
tensor (training/augment.py's job to build) plus a (2, temporal_window)
opponent army/land history array -- mirrors training/network.py's
`net(tensor) -> (logits, value)` interface as closely as the extra temporal
input allows, so training/ppo.py and training/rollout.py only need to swap
which network class + tensor-builder they call, not their own control flow.
"""
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from training.action_space import BUILD_KIND
from training.config import TransformerNetworkConfig

# Mirrors training/network.py's _BUILD_LOGIT_INIT_BIAS exactly -- see reason
# #3 in the module docstring.
_BUILD_LOGIT_INIT_BIAS = -6.0


class MultiHeadSelfAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_head: int, *, key):
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        k1, k2, k3, k4 = jrandom.split(key, 4)
        self.q_proj = eqx.nn.Linear(d_model, d_model, key=k1)
        self.k_proj = eqx.nn.Linear(d_model, d_model, key=k2)
        self.v_proj = eqx.nn.Linear(d_model, d_model, key=k3)
        self.out_proj = eqx.nn.Linear(d_model, d_model, key=k4)

    def __call__(self, x):
        """x: (seq_len, d_model) -> (seq_len, d_model)"""
        seq_len = x.shape[0]
        q = jax.vmap(self.q_proj)(x).reshape(seq_len, self.n_head, self.head_dim)
        k = jax.vmap(self.k_proj)(x).reshape(seq_len, self.n_head, self.head_dim)
        v = jax.vmap(self.v_proj)(x).reshape(seq_len, self.n_head, self.head_dim)

        q = jnp.transpose(q, (1, 0, 2))
        k = jnp.transpose(k, (1, 0, 2))
        v = jnp.transpose(v, (1, 0, 2))

        scale = math.sqrt(self.head_dim)
        attn = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / scale
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.matmul(attn, v)
        out = jnp.transpose(out, (1, 0, 2)).reshape(seq_len, -1)
        return jax.vmap(self.out_proj)(out)


class SelfAttentionLayer(eqx.Module):
    """Pre-norm transformer block: LN -> MHSA -> residual -> LN -> FFN -> residual."""
    norm1: eqx.nn.LayerNorm
    attn: MultiHeadSelfAttention
    norm2: eqx.nn.LayerNorm
    ff_linear1: eqx.nn.Linear
    ff_linear2: eqx.nn.Linear

    def __init__(self, d_model: int, n_head: int, ff_factor: int, *, key):
        k1, k2, k3 = jrandom.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, key=k1)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.ff_linear1 = eqx.nn.Linear(d_model, ff_factor * d_model, key=k2)
        self.ff_linear2 = eqx.nn.Linear(ff_factor * d_model, d_model, key=k3)

    def __call__(self, x):
        x = x + self.attn(jax.vmap(self.norm1)(x))
        h = jax.vmap(self.norm2)(x)
        h = jax.nn.silu(jax.vmap(self.ff_linear1)(h))
        h = jax.vmap(self.ff_linear2)(h)
        return x + h


class TemporalEncoder(eqx.Module):
    """Encodes (2, temporal_window) opponent army/land history into 2 tokens."""
    army_l1: eqx.nn.Linear
    army_l2: eqx.nn.Linear
    land_l1: eqx.nn.Linear
    land_l2: eqx.nn.Linear

    def __init__(self, embed_dim: int, temporal_window: int, hidden: int, *, key):
        k1, k2, k3, k4 = jrandom.split(key, 4)
        self.army_l1 = eqx.nn.Linear(temporal_window, hidden, key=k1)
        self.army_l2 = eqx.nn.Linear(hidden, embed_dim, key=k2)
        self.land_l1 = eqx.nn.Linear(temporal_window, hidden, key=k3)
        self.land_l2 = eqx.nn.Linear(hidden, embed_dim, key=k4)

    def __call__(self, temporal_data):
        """temporal_data: (2, temporal_window) -> (2, embed_dim)"""
        army_hist, land_hist = temporal_data[0] / 50.0, temporal_data[1] / 50.0
        army_token = self.army_l2(jax.nn.silu(self.army_l1(army_hist)))
        land_token = self.land_l2(jax.nn.silu(self.land_l1(land_hist)))
        return jnp.stack([army_token, land_token])


def _unpatchify_policy(patch_logits: jnp.ndarray, gp: int, M: int, num_cell_actions: int) -> jnp.ndarray:
    """(n_patches, num_cell_actions*M*M) -> (p, p, num_cell_actions), CELL-MAJOR
    (kind fastest-varying), matching training/action_space.py's flat-index
    convention. See module docstring point 1 -- this is the single most
    important correctness line in this file."""
    p = gp * M
    x = patch_logits.reshape(gp, gp, num_cell_actions, M, M)
    # axes in: (block_row=0, block_col=1, kind=2, within_row=3, within_col=4)
    # axes out: (block_row, within_row, block_col, within_col, kind)
    x = x.transpose(0, 3, 1, 4, 2)
    return x.reshape(p, p, num_cell_actions)


class TransformerPolicyValueNetwork(eqx.Module):
    embedder: eqx.nn.Linear
    value_token: jnp.ndarray
    pos_encoding: jnp.ndarray
    transformer_layers: list
    norm_out: eqx.nn.LayerNorm
    policy_head: eqx.nn.Linear
    pass_head: eqx.nn.Linear
    value_linear1: eqx.nn.Linear
    value_linear2: eqx.nn.Linear
    temporal_encoder: TemporalEncoder
    temporal_type_embed: jnp.ndarray

    grid_size: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    n_channels: int = eqx.field(static=True)
    num_cell_actions: int = eqx.field(static=True)

    def __init__(self, key: jnp.ndarray, cfg: TransformerNetworkConfig = TransformerNetworkConfig()):
        self.grid_size = cfg.grid_size
        self.patch_size = cfg.patch_size
        self.n_channels = cfg.n_channels
        self.num_cell_actions = cfg.num_cell_actions

        assert cfg.grid_size % cfg.patch_size == 0, \
            f"grid_size ({cfg.grid_size}) must be divisible by patch_size ({cfg.patch_size})"
        gp = cfg.grid_size // cfg.patch_size

        keys = jrandom.split(key, cfg.depth + 8)

        patch_dim = cfg.n_channels * cfg.patch_size * cfg.patch_size
        self.embedder = eqx.nn.Linear(patch_dim, cfg.embed_dim, key=keys[0])

        n_patches = gp * gp
        n_temporal_tokens = 2  # army + land
        n_tokens = n_patches + 1 + n_temporal_tokens  # patches + value + temporal
        self.value_token = jrandom.normal(keys[1], (1, cfg.embed_dim)) * 0.02
        self.pos_encoding = jrandom.truncated_normal(
            keys[2], -2.0, 2.0, (n_tokens, cfg.embed_dim)
        ) * 0.1

        self.transformer_layers = [
            SelfAttentionLayer(cfg.embed_dim, cfg.n_head, cfg.ff_factor, key=keys[3 + i])
            for i in range(cfg.depth)
        ]
        self.norm_out = eqx.nn.LayerNorm(cfg.embed_dim)

        self.policy_head = eqx.nn.Linear(
            cfg.embed_dim, cfg.num_cell_actions * cfg.patch_size * cfg.patch_size, key=keys[3 + cfg.depth]
        )
        # BUILD-channel bias suppression (module docstring point 3). policy_head's
        # bias is laid out (num_cell_actions*M*M,); a patch's local (kind, wr, wc)
        # position kind==BUILD_KIND spans indices [BUILD_KIND*M*M : (BUILD_KIND+1)*M*M).
        M = cfg.patch_size
        build_slice = slice(BUILD_KIND * M * M, (BUILD_KIND + 1) * M * M)
        new_bias = self.policy_head.bias.at[build_slice].set(_BUILD_LOGIT_INIT_BIAS)
        self.policy_head = eqx.tree_at(lambda l: l.bias, self.policy_head, new_bias)

        self.pass_head = eqx.nn.Linear(cfg.embed_dim, 1, key=keys[4 + cfg.depth])
        # Value head: mirrors training/network.py's CNN exactly (Linear ->
        # ReLU -> Linear(,1)), applied to the value/CLS token instead of a
        # global-average-pool. Unbounded output (no activation) -- matches
        # composite_reward_fn's discounted return not being tightly bounded.
        self.value_linear1 = eqx.nn.Linear(cfg.embed_dim, cfg.value_hidden, key=keys[5 + cfg.depth])
        self.value_linear2 = eqx.nn.Linear(cfg.value_hidden, 1, key=keys[6 + cfg.depth])
        self.temporal_encoder = TemporalEncoder(
            cfg.embed_dim, cfg.temporal_window, cfg.temporal_hidden, key=keys[7 + cfg.depth]
        )
        self.temporal_type_embed = jrandom.normal(keys[7 + cfg.depth], (2, cfg.embed_dim)) * 0.02

    def __call__(self, obs_tensor: jnp.ndarray, temporal_data: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """obs_tensor: (n_channels, grid_size, grid_size) float32, already
        normalized+augmented (training/augment.py). temporal_data: (2,
        temporal_window) raw opponent army/land history.

        Returns:
            logits: (grid_size*grid_size*num_cell_actions + 1,) flat raw logits,
                cell-major (matches training/action_space.py), plus one
                trailing global pass logit.
            value: () scalar.
        """
        p = self.grid_size
        M = self.patch_size
        gp = p // M

        # Patchify: (C, p, p) -> (n_patches, C*M*M). Purely an internal
        # tokenization choice -- self-consistent with the unpatchify below
        # via matching patch index i = block_row*gp + block_col, doesn't need
        # to match any external convention (see module docstring point 1).
        x = obs_tensor.reshape(self.n_channels, gp, M, gp, M)
        x = x.transpose(1, 3, 0, 2, 4).reshape(gp * gp, -1)
        x = jax.vmap(self.embedder)(x)

        temporal_tokens = self.temporal_encoder(temporal_data) + self.temporal_type_embed
        x = jnp.concatenate([self.value_token, temporal_tokens, x], axis=0)
        x = x + self.pos_encoding

        for layer in self.transformer_layers:
            x = layer(x)
        x = jax.vmap(self.norm_out)(x)

        value_embedding = x[0]
        patch_embeddings = x[3:]  # skip value + 2 temporal tokens

        patch_logits = jax.vmap(self.policy_head)(patch_embeddings)  # (n_patches, num_cell_actions*M*M)
        cell_logits = _unpatchify_policy(patch_logits, gp, M, self.num_cell_actions)  # (p, p, num_cell_actions)
        cell_logits_flat = cell_logits.reshape(p * p * self.num_cell_actions)

        pass_logit = self.pass_head(value_embedding)  # (1,)
        logits = jnp.concatenate([cell_logits_flat, pass_logit])

        value = self.value_linear2(jax.nn.relu(self.value_linear1(value_embedding)))[0]

        return logits, value
