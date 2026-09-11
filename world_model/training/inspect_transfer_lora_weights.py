#!/usr/bin/env python3
"""Compare frozen base weights between official Transfer2.5 and a LoRA export.

This is a read-only diagnostic helper for the local Transfer2.5 instance-LoRA
smoke test.  It intentionally keeps only a few tensors from the official
checkpoint in memory before opening the adapted checkpoint.
"""

from __future__ import annotations

import argparse
import gc

import torch


TARGET_SUFFIXES = (
    "x_embedder.proj.1.weight",
    "t_embedder.1.linear_1.weight",
    "blocks.0.self_attn.q_proj.weight",
    "blocks.0.self_attn.k_proj.weight",
    "blocks.0.mlp.layer1.weight",
    "blocks.10.self_attn.q_proj.weight",
)


def load_state(path: str) -> dict[str, object]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(obj)!r}")
    return obj


def find_key(state: dict[str, object], suffix: str) -> str | None:
    exact = [k for k in state if k == f"net.{suffix}" or k == suffix]
    if exact:
        return exact[0]
    matches = [k for k in state if k.endswith(suffix) and "lora_" not in k]
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official")
    parser.add_argument("adapted")
    args = parser.parse_args()

    official = load_state(args.official)
    print(f"official_keys={len(official)}")
    print("official_first_keys=", list(official)[:8])

    snapshots: dict[str, torch.Tensor] = {}
    official_keys: dict[str, str] = {}
    for suffix in TARGET_SUFFIXES:
        key = find_key(official, suffix)
        print(f"official[{suffix}]={key}")
        if key is not None and isinstance(official[key], torch.Tensor):
            snapshots[suffix] = official[key].detach().float().clone()
            official_keys[suffix] = key

    del official
    gc.collect()

    adapted = load_state(args.adapted)
    print(f"adapted_keys={len(adapted)}")
    print("adapted_first_keys=", list(adapted)[:8])

    for suffix, ref in snapshots.items():
        key = find_key(adapted, suffix)
        if key is None or not isinstance(adapted[key], torch.Tensor):
            print(f"COMPARE {suffix}: adapted key missing")
            continue
        cur = adapted[key].detach().float()
        if cur.shape != ref.shape:
            print(f"COMPARE {suffix}: shape {tuple(ref.shape)} -> {tuple(cur.shape)}")
            continue
        diff = cur - ref
        ref_norm = torch.linalg.vector_norm(ref).item()
        diff_norm = torch.linalg.vector_norm(diff).item()
        rel = diff_norm / max(ref_norm, 1e-12)
        print(
            f"COMPARE {suffix}: max_abs={diff.abs().max().item():.6g} "
            f"mean_abs={diff.abs().mean().item():.6g} rel_l2={rel:.6g}"
        )

    lora_keys = [k for k in adapted if "lora_A" in k or "lora_B" in k]
    print(f"lora_keys={len(lora_keys)}")
    for key in lora_keys[:8]:
        value = adapted[key]
        if isinstance(value, torch.Tensor):
            x = value.detach().float()
            print(
                f"LORA {key}: shape={tuple(x.shape)} "
                f"mean_abs={x.abs().mean().item():.6g} max_abs={x.abs().max().item():.6g}"
            )

    for branch in ("net.", "net_ema."):
        branch_b = [
            adapted[k].detach().float()
            for k in adapted
            if k.startswith(branch) and "lora_B" in k and isinstance(adapted[k], torch.Tensor)
        ]
        if not branch_b:
            print(f"LORA_B_SUMMARY {branch}: no tensors")
            continue
        nonzero_tensors = sum(int(torch.count_nonzero(x).item() > 0) for x in branch_b)
        total_nonzero = sum(int(torch.count_nonzero(x).item()) for x in branch_b)
        max_abs = max(float(x.abs().max().item()) for x in branch_b)
        mean_abs = sum(float(x.abs().mean().item()) for x in branch_b) / len(branch_b)
        print(
            f"LORA_B_SUMMARY {branch}: tensors={len(branch_b)} "
            f"nonzero_tensors={nonzero_tensors} total_nonzero={total_nonzero} "
            f"mean_abs={mean_abs:.6g} max_abs={max_abs:.6g}"
        )


if __name__ == "__main__":
    main()
