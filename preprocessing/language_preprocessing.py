"""
Frozen natural-language embeddings for each gesture (DistilRoBERTa).

Writes ``<out-dir>/language_embedding_table.npy`` — float32 array of shape (len(GESTURES), D)
with rows in ``GESTURES`` order. Used by ``preprocessing_kinematics.py`` to attach
``language.npy`` per session.

Run after ``sessions/`` / HDF5 layout exists; safe to re-run when phrases or model change.
Typically run **before** ``preprocessing_kinematics.py`` (same ``--out-dir``).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "preprocessed_sessions"

BERT_MODEL = "distilroberta-base"
LANG_TABLE_NAME = "language_embedding_table.npy"

GESTURES: tuple[str, ...] = (
    "grasp",
    "index_pick",
    "press",
    "single_finger",
    "spray",
)
GESTURE_TO_LANGUAGE: dict[str, str] = {
    "grasp": "grasp the cup",
    "index_pick": "pick with the index finger",
    "press": "press the button",
    "single_finger": "single finger movement",
    "spray": "spray",
}
LANGUAGES: list[str] = [GESTURE_TO_LANGUAGE[g] for g in GESTURES]


def parse_recording_name(stem: str) -> tuple[str, int]:
    """``grasp_1`` -> (``grasp``, 1). Validates gesture against ``GESTURE_TO_LANGUAGE``."""
    m = re.match(r"^(.+)_(\d+)$", stem)
    if not m:
        raise ValueError(f"expected {{gesture}}_{{session}}, got {stem!r}")
    g, s = m.group(1), int(m.group(2))
    if g not in GESTURE_TO_LANGUAGE:
        raise ValueError(f"unknown gesture {g!r} in {stem!r}")
    return g, s


def compute_language_embeddings(
    model_name: str,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    with torch.no_grad():
        tokens = tokenizer(
            LANGUAGES,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        hidden = model(**tokens).last_hidden_state
        attn = tokens["attention_mask"].unsqueeze(-1).float()
        emb = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
    dim = int(emb.shape[1])
    return emb.cpu().numpy().astype(np.float32), dim


def main() -> None:
    p = argparse.ArgumentParser(description="Build language_embedding_table.npy (frozen DistilRoBERTa).")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="preprocessed_sessions root (default: <repo>/preprocessed_sessions)",
    )
    p.add_argument("--model", type=str, default=BERT_MODEL, help="HuggingFace model id")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device for embedding pass (default: cpu)",
    )
    args = p.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Language embeddings ({args.model}, frozen, device={device})...")
    lang_emb_table, _dim = compute_language_embeddings(args.model, device)
    if lang_emb_table.shape[0] != len(GESTURES):
        raise RuntimeError("internal: embedding row count must match GESTURES")

    table_path = out_dir / LANG_TABLE_NAME
    np.save(table_path, lang_emb_table)
    print(f"Wrote {table_path}  shape={lang_emb_table.shape}")


if __name__ == "__main__":
    main()
