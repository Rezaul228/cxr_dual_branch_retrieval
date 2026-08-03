# Dual-Branch Multimodal Chest X-ray Retrieval

Code accompanying the Scientific Reports manuscript on dual-branch (Synergy / Difference) multimodal retrieval for chest radiographs and radiology reports.

The model jointly encodes each image–report pair with hierarchical gated co-attention and Global-to-Local Feedback, then ranks cases via cosine similarity over cached embeddings.

## Repository contents

| Path | Description |
|------|-------------|
| `models.py` | Image/text encoders, Synergy and Difference branches, hierarchical co-attention, losses |
| `data_loader.py` | Shard dataset loader and tokenizer utilities |
| `train.py` | Training with synergy / main / orthogonal loss mix |
| `evaluate.py` | Cross-modal Recall@K and MRR evaluation |
| `measure_efficiency.py` | Parameter, FLOP, latency, and memory measurements |
| `ablation_study.py` | Architecture ablation experiments |
| `evaluate_orthogonal_ablation.py` | Orthogonal-loss ablation comparison |
| `visualize.py` | Training curves, retrieval examples, ablation plots |
| `analysis/run_mimic_duplicate_analysis.py` | MIMIC cross-split near-duplicate audit |
| `analysis/run_indiana_duplicate_analysis.py` | Indiana (original-only) duplicate audit |
| `data_processing/` | MIMIC-CXR and Indiana/OpenI → shard preprocessing |
| `checkpoints/` | Released pretrained weights for the reported models |
| `config.py` | Dataset and training hyperparameters |
| `paths.py` | Data and output path configuration |

## Requirements

- Python 3.9+
- CUDA-capable GPU recommended for training

```bash
pip install -r requirements.txt
```

Optional for FLOP counting and baseline comparisons in `measure_efficiency.py`:

```bash
pip install fvcore torchvision transformers
```

## Data access (important)

This repository does **not** redistribute MIMIC-CXR images or reports. Access is restricted under PhysioNet credentialed use.

1. Complete the required PhysioNet training and data use agreement for [MIMIC-CXR](https://physionet.org/content/mimic-cxr/).
2. Obtain Indiana University Chest X-ray data from its public source if using that corpus.
3. Preprocess images and reports into shard pickles with the expected layout (see below).
4. Point this code at your processed data root.

### Expected processed layout

```text
$CXR_DATA_ROOT/
  mimic_shards/                 # or aug_indiana_extended/, indiana_shards/, ...
    metadata.pkl                # includes tokenizer and vocab metadata
    train/
      shard_000.pkl
      ...
    val/
      ...
    test/
      ...
```

Each shard pickle contains dictionaries with keys `images`, `captions`, and `study_ids`.

### Environment configuration

```bash
export CXR_DATA_ROOT=/path/to/all_processed_data
export CXR_DATASET_MODE=aug_indiana_extended   # or mimic_shards, indiana_shards, ...
export CXR_OUTPUTS_DIR=outputs
export CXR_SAVED_MODELS_DIR=saved_models
```

Alternatively edit `config.py` (`DATASET_MODE`) and `paths.py` (`DATA_ROOT`).

## Quick start

### Train

```bash
python train.py --experiment_name dual_branch_v1
```

Useful flags: `--batch_size`, `--learning_rate`, `--epochs`, `--train_samples`, `--val_samples`, `--device`, `--seed`, `--synergy_weight`, `--main_weight`, `--ortho_weight`.

Reported loss mix defaults: synergy 0.64, main 0.20, orthogonal 0.15.

### Evaluate a checkpoint

```bash
python evaluate_checkpoint.py \
  --model-path saved_models/dual_branch_v1/export/model_weights.pth \
  --batch-size 64
```

### Efficiency / complexity table

```bash
python measure_efficiency.py --ours-only
python measure_efficiency.py              # includes ResNet-50 and ClinicalBERT baselines
```

### Ablations

```bash
python ablation_study.py \
  --model-path checkpoints/aug_indiana_full_branch_v4_model_weights.pth

python evaluate_orthogonal_ablation.py \
  --full-model checkpoints/aug_indiana_full_branch_v4_model_weights.pth \
  --no-ortho-model checkpoints/aug_indiana_no_ortho_branch_v4_model_weights.pth
```

### Released checkpoints

See `checkpoints/README.md`. Example:

```bash
python evaluate_checkpoint.py \
  --model-path checkpoints/aug_indiana_full_branch_v4_model_weights.pth
```

### Split-integrity / duplicate analysis

```bash
python analysis/run_mimic_duplicate_analysis.py \
  --base-dir "$CXR_DATA_ROOT/mimic_shards" \
  --out-dir analysis_outputs/mimic \
  --meta-csv /path/to/mimic-cxr-2.0.0-metadata.csv \
  --filtered-csv /path/to/filtered_metadata.csv \
  --reports-root /path/to/mimic-cxr-reports

python analysis/run_indiana_duplicate_analysis.py \
  --base-dir "$CXR_DATA_ROOT/aug_indiana_extended" \
  --out-dir analysis_outputs/indiana \
  --meta-csv /path/to/indiana_study_metadata.csv
```

## Inference note

Each known image–report pair is processed once through both branches to produce image-side and text-side embeddings. Co-attention is not re-run per query–candidate comparison; ranking uses cosine similarity over cached embeddings. Embeddings require both modalities as input.

## Code availability (for manuscript)

Use the following once the Zenodo DOI is issued (fill placeholders):

> The source code, preprocessing scripts, evaluation utilities, and released model
> weights supporting this study are available at
> https://github.com/Rezaul228/cxr_dual_branch_retrieval
> (version v1.0.0) and archived at Zenodo
> (DOI: https://doi.org/10.5281/zenodo.XXXXXXX).
> The repository is released under the MIT License. MIMIC-CXR images and reports
> are not redistributed; authorized users must obtain them from PhysioNet.

See `ZENODO.md` for steps to mint the Zenodo DOI after enabling GitHub integration.

## Citation

If you use this code, please cite the associated Scientific Reports article (DOI to be added upon publication) and this software deposit (Zenodo DOI).

## License

See `LICENSE`. MIMIC-CXR data remain subject to PhysioNet terms and are not included here.

## Contact

For code-related questions, contact the corresponding author listed in the manuscript.
