"""Critical correctness check for training/network_transformer.py (per
~/.claude/plans/drifting-questing-babbage.md): verify the policy output
flattens to the SAME (row, col, kind) -> flat-index mapping that
training/action_space.py's decode_action_index/compute_legal_action_mask
assume. Tests _unpatchify_policy for every single (patch, kind,
within-patch-position) combination, not just a spot check -- a wrong
transpose would still "work" (train, produce finite logits) while silently
scrambling every downstream index, exactly the kind of bug that looks like
"the policy is being weird" for weeks rather than a clean failure.

Run directly: python -m training.test_network_transformer
"""
import numpy as np
import jax.numpy as jnp

from training.action_space import decode_action_index, NUM_CELL_ACTIONS, BUILD_KIND
from training.network_transformer import _unpatchify_policy


def test_cell_major_decode_roundtrip():
    p, M = 9, 3  # grid_size, patch_size -- p must be divisible by M
    gp = p // M
    n_patches = gp * gp
    K = NUM_CELL_ACTIONS

    failures = []
    total = 0
    for block_row in range(gp):
        for block_col in range(gp):
            patch_idx = block_row * gp + block_col
            for kind in range(K):
                for wr in range(M):
                    for wc in range(M):
                        total += 1
                        patch_logits = np.zeros((n_patches, K * M * M), dtype=np.float32)
                        # policy_head's per-patch output is interpreted (in
                        # _unpatchify_policy) as reshape(..., K, M, M) -- so
                        # local flat index within a patch is kind*M*M + wr*M + wc.
                        local_flat = kind * M * M + wr * M + wc
                        patch_logits[patch_idx, local_flat] = 1000.0

                        out = _unpatchify_policy(jnp.asarray(patch_logits), gp, M, K)  # (p, p, K)
                        argmax_idx = int(jnp.argmax(out.reshape(-1)))

                        expected_row = block_row * M + wr
                        expected_col = block_col * M + wc
                        expected_pass_field = 2 if kind == BUILD_KIND else 0
                        decoded = decode_action_index(jnp.asarray(argmax_idx), p, p)
                        pass_field, row, col, direction, split = (int(x) for x in decoded)

                        if pass_field != expected_pass_field or row != expected_row or col != expected_col:
                            failures.append(
                                f"patch=({block_row},{block_col}) kind={kind} within=({wr},{wc}): "
                                f"expected (row={expected_row},col={expected_col},pass={expected_pass_field}), "
                                f"got decode(row={row},col={col},pass={pass_field})"
                            )
                            continue

                        if kind != BUILD_KIND:
                            expected_dir = kind if kind < 4 else kind - 4
                            expected_split = 1 if kind >= 4 else 0
                            if direction != expected_dir or split != expected_split:
                                failures.append(
                                    f"patch=({block_row},{block_col}) kind={kind}: "
                                    f"expected dir={expected_dir} split={expected_split}, "
                                    f"got dir={direction} split={split}"
                                )

    print(f"tested {total} (patch, kind, within-patch-position) combinations")
    if failures:
        print(f"FAILED: {len(failures)} mismatches")
        for f in failures[:20]:
            print("  " + f)
        raise AssertionError(f"{len(failures)} cell-major decode mismatches")
    print("ALL PASSED -- cell-major transpose is correct")


def test_forward_pass_shapes():
    import jax.random as jrandom
    from training.config import TransformerNetworkConfig
    from training.network_transformer import TransformerPolicyValueNetwork

    cfg = TransformerNetworkConfig()
    key = jrandom.PRNGKey(0)
    net = TransformerPolicyValueNetwork(key, cfg)
    obs = jrandom.normal(key, (cfg.n_channels, cfg.grid_size, cfg.grid_size))
    temporal = jrandom.normal(key, (2, cfg.temporal_window))
    logits, value = net(obs, temporal)
    expected_logits = cfg.grid_size * cfg.grid_size * cfg.num_cell_actions + 1
    assert logits.shape == (expected_logits,), f"logits shape {logits.shape} != ({expected_logits},)"
    assert value.shape == (), f"value shape {value.shape} != ()"
    print(f"forward pass OK: logits={logits.shape}, value={float(value):.4f}")


if __name__ == "__main__":
    test_cell_major_decode_roundtrip()
    test_forward_pass_shapes()
