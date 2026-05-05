# KappaPlace

**KappaPlace** is a research framework for visual place recognition that models uncertainty in learned embeddings to improve performance and provide calibrated confidence scores under challenging real-world conditions. The framework builds on [CosPlace](https://github.com/gmberton/CosPlace)-style training and adds learned per-descriptor uncertainty so can evaluate retrieval quality together with **calibration** (expected calibration error, ECE) under challenging conditions.

## Features

- **Config-first CLI**: JSON defaults (`--config`) merged with command-line overrides (see `configs/parser.py`).
- **Models**: `cosplace` (train from scratch) and `cosplace_pretrained` (backbone weights from `torch.hub`).
- **Uncertainty modes** (`--model_mode uncertainty`): variance heads (`--var_head_type`), optional uncertainty losses (`gaussian_nll`, `vmf`), combined with CosFace-style supervision (`--losses`).
- **Evaluation** (`eval.py`): recalls @K, mAP, ECE variants, uncertainty statistics, optional W&B logging and plots (`--save_plots`).
- **Optional logging**: [Weights & Biases](https://wandb.ai) via `--use_wandb`.

## Optional: third-party submodules

Git submodules under `third_party/` ([CosPlace](https://github.com/gmberton/CosPlace), [SUE](https://github.com/MubarizZaffar/SUE), [STUN](https://github.com/ramdrop/stun)) are vendored for reference or comparisons. 

## Data layout

1. Point `data_folder` (and optionally `local_data_folder` for Colab) in your config to the directory that contains your datasets.
2. Each **entry** in the config JSON lists `name`, `src_rel`, and `dst_rel` so the code can resolve `train` / `validation` / `test` folders under that root (see `configs/datasets.json` for examples such as SF-XL, Pittsburgh-30k, MSLS, Nordland, St Lucia, and Amstertime-style benchmarks).
3. Evaluation expects query folders named `queries`, `queries_`*, or `query`* under the test (or validation) split.

Adjust paths in a **local** copy of the config file rather than committing secrets or cluster-specific absolute paths.

## Configuration

- **Default config file**: `configs/datasets.json` (override with `--config path/to/your.json`).
- **Common CLI flags** (non-exhaustive):
  - **Run & I/O**: `--data_folder`, `--logs_folder`, `--device`, `--num_workers`, `--colab`; `--dry_run` (debug: skips file writes, prints what would happen)
  - **Model**: `--method`, `--backbone`, `--descriptors_dimension`, `--image_size`, `--resume_model`, `--resume_train`
  - **Uncertainty**: `--model_mode`, `--uncertainty_loss`, `--var_head_type`, `--uncertainty_lambda`, `--freeze_model`
  - **Training**: `--batch_size`, `--epochs_num`, `--lr`, `--losses`, `--early_stop_metric`, `--patience`
  - **Eval**: `--datasets_type`, `--recall_values`, `--infer_batch_size`, `--save_plots`, `--only_recalls`

Run `python train.py --help` or `python eval.py --help` for the full list.

### Minimal examples

Train (after editing `data_folder` and dataset entries in your JSON):

```bash
python train.py --config configs/datasets.json --model_mode uncertainty --losses ce,uncertainty --groups_num 8
```

A one-step sanity check (one training iteration then exit):

```bash
python train.py --config configs/datasets.json --model_mode uncertainty --losses ce,uncertainty --groups_num 8\
  --dry_run --num_workers 0
```

Evaluate a checkpoint (`--model_mode` must match how the weights were trained; replace the checkpoint path). For UTM-based metrics and writing `recalls.txt` on benchmarks that embed pose in paths (e.g. Pittsburgh), add `**--use_labels**`.

```bash
python eval.py --config configs/datasets.json \
  --resume_model path/to/best_model.pth \
  --model_mode uncertainty \
  --datasets pitts30k \
  --datasets_type test \
  --use_labels
```

## Project layout (high level)


| Path            | Role                                                                                                |
| --------------- | --------------------------------------------------------------------------------------------------- |
| `train.py`      | Training loop, CosFace classifiers                                                                  |
| `eval.py`       | Retrieval metrics, ECE, uncertainty analysis, optional W&B figures                                  |
| `configs/`      | `parser.py` (CLI), JSON configs                                                                     |
| `models/`       | Model factory (`get_model.py`), uncertainty wrapper (`model_mode.py`), CosPlace-uncertainty network |
| `data/`         | Datasets, paths                                                                                     |
| `losses/`       | CosFace, Gaussian / vMF uncertainty losses                                                          |
| `eval_metrics/` | ECE, baselines, visualizations                                                                      |
| `notebooks/`    | Exploratory / evaluation notebooks                                                                  |


## License

Released under the MIT License — see [LICENSE](LICENSE).

## Citation

If you use this code in research, please cite the accompanying paper when available, and consider citing [CosPlace](https://github.com/gmberton/CosPlace) and any baseline methods you compare against (e.g., SUE, STUN) as appropriate.
