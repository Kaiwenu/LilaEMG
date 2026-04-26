# LilaEMG

Language-informed EMG teleoperation experiments: train a **FiLM-conditioned latent model** from synchronized sEMG and hand kinematics, in three stages (encoder/decoder on kinematics → EMG teacher matching → optional fine-tune). The training stack lives at the repository root; the **`lila/`** subtree is the original LILA research codebase (configs, Lightning, Franka tooling) and is separate from the EMG pipeline described here.

## Layout

| Path | Role |
|------|------|
| `data/` | Raw recordings: `*.hdf5` with `emg2pose/timeseries` (`time`, `joint_angles` 20-D, `emg` 8-D). Names like `grasp_1.hdf5`. |
| `sessions/` | Per-recording exports and sEMG processing (`semg.npy` → filtered → z-scored). |
| `preprocessed_sessions/` | Windowed EMG, PCA hand state/velocity, language vectors, `pca_joint_angles.joblib`. Produced by `preprocessing/`. |
| `checkpoints/` | `train_teleop.py` outputs: `stage{N}/run_<timestamp>/` with `metrics.csv`, `best.pt`, epoch checkpoints. |
| `preprocessing/` | All HDF5 → training tensor scripts plus `run_pipeline.py`. |
| `train_teleop.py` | Multi-stage training CLI. |
| `teleop_model.py` | `LilaTeleopModel` definition. |
| `plot_training_loss.py`, `visualize_stage2_data.py` | Post-hoc plots and Stage 2 diagnostics. |

## Dependencies

Install a recent **PyTorch** build for your platform (CPU or CUDA). Other Python packages used by training and preprocessing:

```bash
pip install numpy scipy scikit-learn h5py joblib transformers torch
```

Optional: **`matplotlib`** (training writes `training_loss.png` when available).

There is no checked-in `requirements.txt`; align PyTorch with your [CUDA driver](https://pytorch.org/) if you use a GPU. If CUDA initialization fails at import time, train with `--device cpu`.

## Preprocessing

End-to-end (from repository root):

```bash
python preprocessing/run_pipeline.py
python preprocessing/run_pipeline.py --dry-run          # print steps only
python preprocessing/run_pipeline.py --skip-emg-features
```

Default order inside `run_pipeline.py`:

1. **`export_sessions_npy.py`** — HDF5 → `sessions/<stem>/` (`semg.npy`, `joint_angles.npy`, `time.npy`).
2. **`filter_semg_sessions.py`** — Butterworth / notch on `semg.npy` → `semg_filtered.npy`.
3. **`normalize_semg_sessions.py`** — Train-split z-score → `semg_filtered_norm.npy`.
4. **`semg_windowing.py`** — Sliding windows → `preprocessed_sessions/<stem>/emg_windows_full.npy`.
5. **`language_preprocessing.py`** — DistilRoBERTa phrase embeddings → `language_embedding_table.npy`.
6. **`preprocessing_kinematics.py`** — Global PCA on joint windows, velocities, aligned `emg_window.npy`, per-session labels.
7. **`extract_emg_features.py`** — Time-domain features → `emg_features.npy` (default EMG input for training).

Optional: **`--also-flat-emg`** runs **`extract_emg.py`** (flat `emg_npy/*_emg.npy`); not required for `train_teleop.py`.

Shared window parameters live in **`preprocessing/window_utils.py`**.

If you only refresh later stages (e.g. sEMG already normalized), run the corresponding scripts directly with the same `--data-dir` / `--sessions-dir` / `--preprocessed-dir` as in `run_pipeline.py --help`.

## Training (`train_teleop.py`)

**Stages**

1. **Stage 1** — Train FiLM encoder + decoder on `hand_state`, `hand_velocity`, language; loss is velocity MSE. EMG is loaded for optional normalization but not in the Stage 1 loss.
2. **Stage 2** — Freeze encoder and decoder; train the EMG head to match the encoder latent (teacher still uses language).
3. **Stage 3** — Freeze encoder; fine-tune EMG + decoder toward velocity MSE (optional).

**EMG input:** `--emg-input window` (320-D raw windows) or `--emg-input features` (48-D; default).

**Split:** Per gesture, sessions ordered by numeric id: 4 train / 1 val / 1 test when six sessions exist; extras go to train. See `--only-gestures` and `--single-session` in the script docstring.

**Example**

```bash
# Stage 1 (defaults: 30 epochs, preprocessed_sessions/, checkpoints/)
python train_teleop.py --stage 1

python train_teleop.py --stage 2 \
  --resume checkpoints/stage1/run_<timestamp>/best.pt

python train_teleop.py --stage 3 \
  --resume checkpoints/stage2/run_<timestamp>/best.pt
```

Each run creates a new `run_<UTC>/` under `checkpoints/stage{N}/` so prior runs are not overwritten. Use `--device cpu`, `--epochs`, `--batch-size`, `--preprocessed-dir`, `--out-dir`, and `--tag` as needed.

**Plots**

```bash
python plot_training_loss.py --checkpoints-dir checkpoints
```

## Bundled `lila/` package

`lila/` contains the upstream LILA training entrypoint (`lila/train.py`), YAML configs under `lila/conf/`, and dataset/model code under `lila/src/`. It is **not** invoked by `train_teleop.py`; use it when reproducing paper experiments or robot demos from that stack.
