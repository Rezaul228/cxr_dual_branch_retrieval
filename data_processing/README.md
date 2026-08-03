# Data processing (MIMIC-CXR and Indiana / OpenI)

These scripts convert authorized raw chest X-ray corpora into the shard format
expected by the training and evaluation code in the repository root.

**No raw images, reports, or processed shards are included.** Users must obtain
data under the applicable licenses (PhysioNet for MIMIC-CXR; Indiana/OpenI
public release terms).

## Scripts included

| Script | Role |
|--------|------|
| `process_mimic_data.py` | MIMIC-CXR → shards (CLI entry) |
| `mimic_data_loader.py` | MIMIC loading, splitting, tokenization |
| `build_mimic_vocab.py` | Build MIMIC vocabulary JSON |
| `process_indiana_data.py` | Indiana/OpenI → shards (Indiana vocab) |
| `process_indiana_simplified.py` | Indiana/OpenI → shards using MIMIC vocab (cross-dataset) |
| `data_set_loader.py` | Indiana loader used by `process_indiana_data.py` |
| `data_set_loader_simplified.py` | Indiana loader used by `process_indiana_simplified.py` |
| `build_indiana_vocab.py` | Build Indiana vocabulary |
| `enhanced_data_loader.py` | Shared medical text cleaning and `EnhancedTokenizer` |
| `pre_augmentation_vocab_builder.py` | Vocabulary preparation before augmentation |
| `adv_aug_config.py` / `adv_aug_image.py` / `adv_aug_text.py` | Image/text augmentation for extended Indiana |
| `create_aug_metadata.py` | Metadata helpers for augmented Indiana shards |

## Expected raw inputs

### MIMIC-CXR (PhysioNet credentialed access)

- Metadata CSV (e.g. `mimic-cxr-2.0.0-metadata.csv`)
- Free-text reports directory
- Chest radiograph images (JPG/PNG as used in your local layout)

### Indiana / OpenI

- Reports CSV
- Projections / image-index CSV
- Image directory

## Typical pipeline

```bash
cd data_processing

# 1) Optional: build vocabulary from raw reports
python build_mimic_vocab.py   # see script --help / paths inside

# 2) Process MIMIC into shards
python process_mimic_data.py \
  --metadata_csv /path/to/mimic-cxr-2.0.0-metadata.csv \
  --reports_dir /path/to/mimic-cxr-reports \
  --images_dir /path/to/mimic-cxr-images \
  --output_dir /path/to/all_processed_data/mimic_shards

# 3) Process Indiana/OpenI (MIMIC-compatible tokenizer for cross-dataset eval)
python process_indiana_simplified.py \
  --reports_csv /path/to/indiana_reports.csv \
  --projections_csv /path/to/indiana_projections.csv \
  --image_dir /path/to/indiana_images \
  --output_dir /path/to/all_processed_data/indiana_shards_zero_shot
```

Exact flags may vary slightly by script; run `python <script.py> --help` where available.

Output layout (required by the model code):

```text
$CXR_DATA_ROOT/<dataset_name>/
  metadata.pkl
  train/shard_*.pkl
  val/shard_*.pkl
  test/shard_*.pkl
```

Then set:

```bash
export CXR_DATA_ROOT=/path/to/all_processed_data
export CXR_DATASET_MODE=mimic_shards   # or aug_indiana_extended, etc.
```

## Notes

- Several scripts still contain local-path defaults from development. Override
  them with CLI arguments or edit path constants before running.
- Augmented Indiana (`aug_indiana_extended`) additionally uses the `adv_aug_*`
  modules after a base Indiana shard build.
- Do not upload PhysioNet credentials, cookies, or any patient-level files to
  public repositories.
