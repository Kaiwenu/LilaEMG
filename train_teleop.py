"""
Train LILA-style EMG teleop model (see teleop_model.LilaTeleopModel).

Stages (from proposal):
  1 — Train action encoder + language-conditioned decoder on vision labels
      (hand_state, hand_velocity, language_embedding). Loss: velocity MSE.
  2 — Freeze encoder & decoder; train EMG net to match encoder latent.
  3 — Freeze encoder only; fine-tune EMG + decoder toward velocity MSE (optional).

Data: preprocessed/preprocessed.npz + preprocessed_meta.json from preprocessing.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from teleop_model import LilaTeleopModel

ROOT = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_DIR = ROOT / "preprocessed"
NPZ_NAME = "preprocessed.npz"
META_NAME = "preprocessed_meta.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_meta(preprocessed_dir: Path) -> dict:
    path = preprocessed_dir / META_NAME
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def recording_key(gesture_id: np.ndarray, session: np.ndarray, idx: int) -> int:
    """Stable int id for (gesture, session) without collisions."""
    return int(gesture_id[idx]) * 10_000 + int(session[idx])


def train_val_indices_by_recording(
    gesture_id: np.ndarray,
    session: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole (gesture, session) groups so adjacent frames don't leak."""
    n = gesture_id.shape[0]
    keys = np.array([recording_key(gesture_id, session, i) for i in range(n)], dtype=np.int64)
    unique = np.unique(keys)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    val_keys = set(unique[:n_val].tolist())
    mask_val = np.array([k in val_keys for k in keys], dtype=bool)
    idx_all = np.arange(n, dtype=np.int64)
    return idx_all[~mask_val], idx_all[mask_val]


class NormStats:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, names: tuple[str, ...]):
        self.mean = mean
        self.std = std
        self.names = names

    def to(self, device: torch.device) -> NormStats:
        return NormStats(self.mean.to(device), self.std.to(device), self.names)

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "names": self.names}

    @classmethod
    def from_state_dict(cls, d: dict) -> NormStats:
        return cls(d["mean"], d["std"], tuple(d["names"]))


def compute_norm_stats(
    z: dict[str, np.ndarray],
    train_idx: np.ndarray,
    fields: tuple[str, ...],
) -> dict[str, NormStats]:
    out: dict[str, NormStats] = {}
    for name in fields:
        arr = np.asarray(z[name][train_idx], dtype=np.float64)
        mean = torch.from_numpy(arr.mean(axis=0)).float()
        std = torch.from_numpy(arr.std(axis=0)).float()
        std = torch.clamp(std, min=1e-6)
        out[name] = NormStats(mean, std, (name,))
    return out


class TeleopNpzDataset(Dataset):
    """Index into in-memory arrays from ``preprocessed.npz`` (compressed npz row access is not mmap-safe)."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        indices: np.ndarray | None,
        norm: dict[str, NormStats] | None,
        device_for_norm: str = "cpu",
    ):
        self.arrays = arrays
        self.indices = indices if indices is not None else np.arange(arrays["hand_state"].shape[0])
        self.norm = norm
        self.dev = torch.device(device_for_norm)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _norm_field(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if self.norm is None or name not in self.norm:
            return x
        st = self.norm[name]
        m = st.mean.to(self.dev)
        s = st.std.to(self.dev)
        return (x.to(self.dev) - m) / s

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        emg = torch.from_numpy(self.arrays["emg_window"][idx].astype(np.float32, copy=False))
        s = torch.from_numpy(self.arrays["hand_state"][idx].astype(np.float32, copy=False))
        v = torch.from_numpy(self.arrays["hand_velocity"][idx].astype(np.float32, copy=False))
        lang = torch.from_numpy(self.arrays["language_embedding"][idx].astype(np.float32, copy=False))
        emg = self._norm_field("emg_window", emg)
        s = self._norm_field("hand_state", s)
        v = self._norm_field("hand_velocity", v)
        return {"emg_window": emg, "hand_state": s, "hand_velocity": v, "language_embedding": lang}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def save_checkpoint(
    path: Path,
    model: LilaTeleopModel,
    optimizer: torch.optim.Optimizer | None,
    stage: int,
    epoch: int,
    meta_train: dict,
    norm_state: dict | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "stage": stage,
        "epoch": epoch,
        "meta_train": meta_train,
        "norm": norm_state,
    }
    torch.save(payload, path)


def load_model_checkpoint(path: Path, model: LilaTeleopModel) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return ckpt


def run_epoch_stage1(
    model: LilaTeleopModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> float:
    if train:
        model.encoder.train()
        model.decoder.train()
    else:
        model.encoder.eval()
        model.decoder.eval()
    total = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            s = batch["hand_state"].to(device)
            v = batch["hand_velocity"].to(device)
            lang = batch["language_embedding"].to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            z = model.encode_action(s, v)
            v_hat = model.decode(s, z, lang)
            loss = F.mse_loss(v_hat, v)
            if train:
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * s.shape[0]
            n += s.shape[0]
    return total / max(n, 1)


def run_epoch_stage2(
    model: LilaTeleopModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> float:
    model.encoder.eval()
    model.decoder.eval()
    model.emg.train(train)
    total = 0.0
    n = 0
    for batch in loader:
        s = batch["hand_state"].to(device)
        v = batch["hand_velocity"].to(device)
        emg = batch["emg_window"].to(device)
        with torch.no_grad():
            z_star = model.encode_action(s, v)
        if train:
            optimizer.zero_grad(set_to_none=True)
            z_emg = model.emg_to_latent(emg)
            loss = F.mse_loss(z_emg, z_star)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                z_emg = model.emg_to_latent(emg)
                loss = F.mse_loss(z_emg, z_star)
        total += float(loss.item()) * s.shape[0]
        n += s.shape[0]
    return total / max(n, 1)


def run_epoch_stage3(
    model: LilaTeleopModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> float:
    model.encoder.eval()
    if train:
        model.emg.train()
        model.decoder.train()
    else:
        model.emg.eval()
        model.decoder.eval()
    total = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            s = batch["hand_state"].to(device)
            v = batch["hand_velocity"].to(device)
            emg = batch["emg_window"].to(device)
            lang = batch["language_embedding"].to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            z = model.emg_to_latent(emg)
            v_hat = model.decode(s, z, lang)
            loss = F.mse_loss(v_hat, v)
            if train:
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * s.shape[0]
            n += s.shape[0]
    return total / max(n, 1)


def freeze_module(m: nn.Module, frozen: bool) -> None:
    for p in m.parameters():
        p.requires_grad_(not frozen)


def build_model(args: argparse.Namespace, synergy_dim: int) -> LilaTeleopModel:
    return LilaTeleopModel(
        synergy_dim=synergy_dim,
        latent_dim=args.latent_dim,
        emg_dim=args.emg_dim,
        language_dim=args.language_dim,
        hidden_dim=args.hidden_dim,
        encoder_hidden_layers=args.encoder_layers,
        emg_hidden_layers=args.emg_layers,
        decoder_film_layers=args.decoder_film_layers,
        decoder_use_language=not args.no_language,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LILA-style EMG teleop model.")
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to load (required after stage 1).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--tag", type=str, default="", help="Subfolder under out-dir.")

    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--emg-dim", type=int, default=320)
    parser.add_argument("--language-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--emg-layers", type=int, default=2)
    parser.add_argument("--decoder-film-layers", type=int, default=2)
    parser.add_argument("--no-language", action="store_true", help="Decoder without FiLM (baseline).")

    args = parser.parse_args()
    set_seed(args.seed)

    pre = args.preprocessed_dir
    npz_path = pre / NPZ_NAME
    if not npz_path.is_file():
        print(f"Missing {npz_path}; run preprocessing.py first.", file=sys.stderr)
        sys.exit(1)

    meta = load_meta(pre)
    synergy_dim = int(meta["n_synergy_components"])
    if args.emg_dim != int(meta.get("emg_window_length", 320)):
        print(
            f"Warning: --emg-dim {args.emg_dim} != meta emg_window_length {meta.get('emg_window_length')}",
            file=sys.stderr,
        )
    if args.language_dim != int(meta.get("language_embedding_dim", 768)):
        print(
            f"Warning: --language-dim vs meta language_embedding_dim mismatch.",
            file=sys.stderr,
        )

    print("Loading preprocessed.npz into RAM (one-time decompress)...", flush=True)
    zfile = np.load(npz_path)
    arrays = {k: zfile[k] for k in zfile.files}
    zfile.close()

    gid = np.asarray(arrays["gesture_id"])
    sess = np.asarray(arrays["session"])
    train_idx, val_idx = train_val_indices_by_recording(gid, sess, args.val_fraction, args.seed)

    norm_fields = ("emg_window", "hand_state", "hand_velocity")
    norm_stats: dict[str, NormStats] | None = None
    if args.normalize:
        raw_stats = compute_norm_stats(arrays, train_idx, norm_fields)
        norm_stats = raw_stats

    train_ds = TeleopNpzDataset(arrays, train_idx, norm_stats, device_for_norm="cpu")
    val_ds = TeleopNpzDataset(arrays, val_idx, norm_stats, device_for_norm="cpu")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model = build_model(args, synergy_dim).to(device)

    if args.stage >= 2:
        if args.resume is None or not args.resume.is_file():
            print("Stage 2/3 requires --resume PATH to a checkpoint from the previous stage.", file=sys.stderr)
            sys.exit(1)
        ckpt_prev = load_model_checkpoint(args.resume, model)
        print(
            f"Loaded weights from {args.resume} "
            f"(saved stage {ckpt_prev.get('stage')}, epoch {ckpt_prev.get('epoch')})"
        )
    elif args.resume is not None and args.resume.is_file():
        ckpt_prev = load_model_checkpoint(args.resume, model)
        print(f"Warm start from {args.resume} (epoch {ckpt_prev.get('epoch')})")

    if args.stage == 2:
        freeze_module(model.encoder, True)
        freeze_module(model.decoder, True)
        freeze_module(model.emg, False)
        params = list(model.emg.parameters())
    elif args.stage == 3:
        freeze_module(model.encoder, True)
        freeze_module(model.decoder, False)
        freeze_module(model.emg, False)
        params = list(model.emg.parameters()) + list(model.decoder.parameters())
    else:
        freeze_module(model.encoder, False)
        freeze_module(model.decoder, False)
        freeze_module(model.emg, False)
        params = list(model.encoder.parameters()) + list(model.decoder.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    run_dir = args.out_dir / (f"stage{args.stage}" + (f"_{args.tag}" if args.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_train = {
        "synergy_dim": synergy_dim,
        "stage": args.stage,
        "preprocessed_dir": str(pre.resolve()),
        "train_samples": int(train_idx.shape[0]),
        "val_samples": int(val_idx.shape[0]),
        "normalize": args.normalize,
        "no_language": args.no_language,
        "args": vars(args),
    }
    norm_state = {k: v.state_dict() for k, v in norm_stats.items()} if norm_stats else None

    best_val = float("inf")
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if args.stage == 1:
            tr = run_epoch_stage1(model, train_loader, optimizer, device, train=True)
            va = run_epoch_stage1(model, val_loader, optimizer, device, train=False)
        elif args.stage == 2:
            tr = run_epoch_stage2(model, train_loader, optimizer, device, train=True)
            va = run_epoch_stage2(model, val_loader, optimizer, device, train=False)
        else:
            tr = run_epoch_stage3(model, train_loader, optimizer, device, train=True)
            va = run_epoch_stage3(model, val_loader, optimizer, device, train=False)

        print(f"epoch {epoch:03d}  train {tr:.6f}  val {va:.6f}")

        ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(ckpt_path, model, optimizer, args.stage, epoch, meta_train, norm_state)

        if va < best_val:
            best_val = va
            save_checkpoint(run_dir / "best.pt", model, optimizer, args.stage, epoch, meta_train, norm_state)

    dt = time.perf_counter() - t0
    print(f"Done stage {args.stage} in {dt:.1f}s. Best val loss {best_val:.6f}. Artifacts in {run_dir}")


if __name__ == "__main__":
    main()
