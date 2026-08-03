# Released checkpoints

Pretrained weights corresponding to the reported dual-branch experiments on
`aug_indiana_extended` (loss mix synergy 0.64 / main 0.20 / orthogonal 0.15,
unless noted).

| File | Description |
|------|-------------|
| `aug_indiana_full_branch_v4_model_weights.pth` | Main paper model (with orthogonal regularization) |
| `aug_indiana_no_ortho_branch_v4_model_weights.pth` | Matched ablation with `ortho_weight=0` |

## Evaluate

From the repository root (after configuring `CXR_DATA_ROOT` and dataset mode):

```bash
python evaluate_checkpoint.py \
  --model-path checkpoints/aug_indiana_full_branch_v4_model_weights.pth \
  --batch-size 64
```

These files contain model parameters only. They do **not** include MIMIC-CXR or
Indiana images/reports. Access to evaluation data remains subject to PhysioNet
and dataset licenses.
