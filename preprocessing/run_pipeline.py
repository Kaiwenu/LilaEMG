"""
End-to-end EMG teleop preprocessing: runs every script in ``preprocessing/`` that belongs
to the training data path (HDF5 → sessions → filtered/normalized sEMG → windowed EMG →
language table → kinematics/PCA → optional time-domain EMG features).

Default order (same defaults as each script when flags are omitted):

1. ``export_sessions_npy.py`` — ``data/*.hdf5`` → ``sessions/<stem>/`` (``semg.npy``, …)
2. ``filter_semg_sessions.py`` — ``semg.npy`` → ``semg_filtered.npy``
3. ``normalize_semg_sessions.py`` — train-split z-score → ``semg_filtered_norm.npy``
4. ``semg_windowing.py`` — → ``preprocessed_sessions/<stem>/emg_windows_full.npy``
5. ``language_preprocessing.py`` — → ``preprocessed_sessions/language_embedding_table.npy``
6. ``preprocessing_kinematics.py`` — PCA, velocity, ``emg_window.npy``, labels, …
7. ``extract_emg_features.py`` — ``emg_window.npy`` → ``emg_features.npy`` (unless ``--skip-emg-features``)

Optional (not required for ``train_teleop.py``):

- ``--also-flat-emg`` — run ``extract_emg.py`` (flat ``*_emg.npy`` under ``emg_npy/``), after export.

``window_utils.py`` is imported by other modules only; it is not executed as a step.

Run from repository root::

    python preprocessing/run_pipeline.py
    python preprocessing/run_pipeline.py --dry-run
    python preprocessing/run_pipeline.py --data-dir /path/to/data --sessions-dir /path/to/sessions
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PY = sys.executable

DEFAULT_DATA = PROJECT_ROOT / "data"
DEFAULT_SESSIONS = PROJECT_ROOT / "sessions"
DEFAULT_PREPROCESSED = PROJECT_ROOT / "preprocessed_sessions"
DEFAULT_FLAT_EMG = PROJECT_ROOT / "emg_npy"


def _run(script: str, argv: list[str], *, dry_run: bool) -> None:
    cmd = [PY, str(HERE / script)] + argv
    label = f"{script} {' '.join(argv)}".strip()
    print(f"\n==> {label}\n")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run full LilaEMG preprocessing pipeline (all preprocessing/*.py stages)."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"HDF5 folder (default: {DEFAULT_DATA})",
    )
    p.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help=f"Per-session npy root (default: {DEFAULT_SESSIONS})",
    )
    p.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=None,
        help=f"Output for windowing + kinematics + language table (default: {DEFAULT_PREPROCESSED})",
    )
    p.add_argument(
        "--skip-emg-features",
        action="store_true",
        help="Do not run extract_emg_features.py (skip emg_features.npy)",
    )
    p.add_argument(
        "--also-flat-emg",
        action="store_true",
        help="After export, also run extract_emg.py → flat *_emg.npy (not used by train_teleop)",
    )
    p.add_argument(
        "--flat-emg-out-dir",
        type=Path,
        default=None,
        help=f"With --also-flat-emg: --out-dir for extract_emg.py (default: {DEFAULT_FLAT_EMG})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned steps only; do not execute subprocesses",
    )
    args = p.parse_args()

    data_dir = args.data_dir.resolve() if args.data_dir is not None else None
    sessions_dir = args.sessions_dir.resolve() if args.sessions_dir is not None else None
    preprocessed_dir = args.preprocessed_dir.resolve() if args.preprocessed_dir is not None else None

    def opt(flag: str, val: Path | None) -> list[str]:
        if val is None:
            return []
        return [flag, str(val)]

    # Only forward paths when the user set them; otherwise each script uses its own defaults.
    export_argv = opt("--data-dir", data_dir) + opt("--out-dir", sessions_dir)
    semg_argv = opt("--data-dir", data_dir) + opt("--sessions-dir", sessions_dir) + opt("--out-dir", preprocessed_dir)
    language_argv = opt("--out-dir", preprocessed_dir)
    kin_argv = opt("--data-dir", data_dir) + opt("--out-dir", preprocessed_dir)
    features_argv = opt("--preprocessed-dir", preprocessed_dir)

    filter_argv = opt("--sessions-dir", sessions_dir)
    norm_argv = opt("--sessions-dir", sessions_dir)

    flat_emg_out = (
        args.flat_emg_out_dir.resolve() if args.flat_emg_out_dir is not None else DEFAULT_FLAT_EMG
    )
    flat_emg_argv = opt("--data-dir", data_dir) + ["--out-dir", str(flat_emg_out)]

    steps: list[tuple[str, list[str]]] = [
        ("export_sessions_npy.py", export_argv),
        ("filter_semg_sessions.py", filter_argv),
        ("normalize_semg_sessions.py", norm_argv),
        ("semg_windowing.py", semg_argv),
        ("language_preprocessing.py", language_argv),
        ("preprocessing_kinematics.py", kin_argv),
    ]
    if not args.skip_emg_features:
        steps.append(("extract_emg_features.py", features_argv))

    if args.dry_run:
        print("Dry run — commands that would execute:\n")

    if args.also_flat_emg:
        steps.insert(1, ("extract_emg.py", flat_emg_argv))

    for script, argv in steps:
        _run(script, argv, dry_run=args.dry_run)

    if args.dry_run:
        print("\n(dry run: no subprocesses started)\n")
    else:
        print("\nPipeline finished.\n")


if __name__ == "__main__":
    main()
