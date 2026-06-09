#!/usr/bin/env python3
"""Inspect a Nemotron CoreML model directory and emit ModelSignatures.json.

Why this exists: the Swift decode loop (Phases 5-7) must NOT hardcode CoreML
tensor names, shapes, or language prompt IDs. This script reads the ground truth
straight from the model and produces a single JSON the app loads at runtime.

It is intentionally dependency-free: it parses the plaintext `model.mil` inside
each `.mlmodelc` (the MIL `func main(...)` signature + return statement) and the
model's `metadata.json`. No coremltools / huggingface_hub required, so it runs
even on a machine that only has Python + the downloaded model.

Usage:
    python3 scripts/inspect_model.py Models/multilingual/2240ms \
        --out NemotronASRPoC/ASR/ModelSignatures.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Modules we try to inspect. decoder_joint is the fused default path;
# decoder + joint are the unfused fallback. preprocessor/encoder are required.
MODULES = ["preprocessor", "encoder", "decoder_joint", "decoder", "joint"]

# tensor<DTYPE, [d0, d1, ...]> name
_TENSOR_ARG = re.compile(r"tensor<\s*(\w+)\s*,\s*\[([^\]]*)\]\s*>\s*(\w+)")
_FUNC_MAIN = re.compile(r"func\s+main<[^>]*>\s*\(")


def _split_top_level(s: str) -> list[str]:
    """Split a comma list ignoring commas inside <> or [] brackets."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        if ch in "<[":
            depth += 1
        elif ch in ">]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return [p.strip() for p in parts if p.strip()]


def _balanced_paren(text: str, open_idx: int) -> str:
    """Return the substring inside the parens starting at `open_idx` ('(')."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
    raise ValueError("unbalanced parentheses in func main signature")


def _parse_arg(arg: str) -> dict | None:
    m = _TENSOR_ARG.search(arg)
    if not m:
        return None
    dtype, shape_raw, name = m.group(1), m.group(2), m.group(3)
    shape = [d.strip() for d in shape_raw.split(",") if d.strip()]
    return {"name": name, "dtype": dtype, "shape": shape}


def _find_output_shape(mil: str, name: str) -> dict:
    """Find the declared tensor type for an output var (last assignment wins)."""
    pat = re.compile(
        r"tensor<\s*(\w+)\s*,\s*\[([^\]]*)\]\s*>\s*" + re.escape(name) + r"\s*="
    )
    matches = list(pat.finditer(mil))
    if not matches:
        return {"name": name, "dtype": None, "shape": None}
    dtype, shape_raw = matches[-1].group(1), matches[-1].group(2)
    shape = [d.strip() for d in shape_raw.split(",") if d.strip()]
    return {"name": name, "dtype": dtype, "shape": shape}


def inspect_module(mlmodelc: Path) -> dict:
    mil_path = mlmodelc / "model.mil"
    mil = mil_path.read_text(encoding="utf-8", errors="replace")

    fm = _FUNC_MAIN.search(mil)
    if not fm:
        raise ValueError(f"no `func main` in {mil_path}")
    args_str = _balanced_paren(mil, fm.end() - 1)
    inputs = [a for a in (_parse_arg(p) for p in _split_top_level(args_str)) if a]

    # Outputs: last `} -> (a, b, c)`
    rets = re.findall(r"\}\s*->\s*\(([^;]*)\)", mil)
    out_names = _split_top_level(rets[-1]) if rets else []
    outputs = [_find_output_shape(mil, n) for n in out_names]

    return {"inputs": inputs, "outputs": outputs}


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit ModelSignatures.json from a Nemotron CoreML dir.")
    ap.add_argument("model_dir", help="e.g. Models/multilingual/2240ms")
    ap.add_argument("--out", default="NemotronASRPoC/ASR/ModelSignatures.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"error: {model_dir} is not a directory", file=sys.stderr)
        return 1

    result: dict = {
        "_generated_by": "scripts/inspect_model.py",
        "source": str(model_dir),
        "modules": {},
    }

    for module in MODULES:
        mlmodelc = model_dir / f"{module}.mlmodelc"
        if not (mlmodelc / "model.mil").is_file():
            continue
        try:
            result["modules"][module] = inspect_module(mlmodelc)
            print(f"  ✓ {module}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  ✗ {module}: {exc}", file=sys.stderr)

    # Fold in the runtime constants + prompt dictionary from metadata.json.
    meta_path = model_dir / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        result["metadata"] = {
            "model": meta.get("model"),
            "sample_rate": meta.get("sample_rate"),
            "mel_features": meta.get("mel_features"),
            "chunk_mel_frames": meta.get("chunk_mel_frames"),
            "pre_encode_cache": meta.get("pre_encode_cache"),
            "total_mel_frames": meta.get("total_mel_frames"),
            "vocab_size": meta.get("vocab_size"),
            "blank_idx": meta.get("blank_idx"),
            "decoder_hidden": meta.get("decoder_hidden"),
            "decoder_layers": meta.get("decoder_layers"),
            "encoder_dim": meta.get("encoder_dim"),
            "cache_channel_shape": meta.get("cache_channel_shape"),
            "cache_time_shape": meta.get("cache_time_shape"),
            "default_prompt_id": meta.get("default_prompt_id"),
        }
        result["prompt_dictionary"] = meta.get("prompt_dictionary", {})
    else:
        print(f"warning: {meta_path} missing; metadata + prompt map omitted", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(result['modules'])} modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
