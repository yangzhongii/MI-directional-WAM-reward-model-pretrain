"""Rebase and validate a copied rigid-v3 cache on an ablation worker.

The canonical manifests contain absolute artifact paths.  A remote worker has
the same run-relative layout under a different repository root, so rsync alone
is insufficient: manifests and cache source signatures must be rebased before
the normal reward script can safely reuse the copied tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mi_reward.data.schema import SuccessReference, TrajectoryExample, read_jsonl
from mi_reward.features.cached_feature_store import (
    CachedFeatureStore,
    feature_bundle_is_cached,
    feature_source_signature,
)


def _rewrite_roots(value: Any, old_root: str, new_root: str) -> Any:
    if isinstance(value, str):
        return value.replace(old_root, new_root)
    if isinstance(value, list):
        return [_rewrite_roots(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_roots(item, old_root, new_root) for key, item in value.items()}
    return value


def rebase_jsonl(path: str | Path, old_root: str | Path, new_root: str | Path) -> int:
    target = Path(path)
    old_value = str(Path(old_root).resolve())
    new_value = str(Path(new_root).resolve())
    records: list[dict[str, Any]] = []
    changed = 0
    for line_no, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Expected a JSON object at {target}:{line_no}.")
        rewritten = _rewrite_roots(item, old_value, new_value)
        changed += int(rewritten != item)
        records.append(rewritten)
    target.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    return changed


def refresh_feature_signatures(
    manifest: str | Path,
    success_refs: str | Path,
    feature_root: str | Path,
) -> dict[str, int]:
    store = CachedFeatureStore(feature_root)
    examples = read_jsonl(manifest, TrajectoryExample)
    refs = read_jsonl(success_refs, SuccessReference)
    missing: list[str] = []
    missing_sources: list[str] = []
    refreshed = 0
    for item in [*examples, *refs]:
        item_id = item.traj_id if isinstance(item, TrajectoryExample) else item.ref_id
        source_paths = [
            *item.frames,
            item.robot_state_path,
            item.object_state_path,
            getattr(item, "relation_path", None),
        ]
        for raw_path in source_paths:
            if raw_path is not None and not Path(raw_path).is_file():
                missing_sources.append(f"{item_id}:{raw_path}")
        signature = feature_source_signature(
            item.frames,
            item.robot_state_path,
            item.object_state_path,
        )
        metadata_path = store.metadata_path_for(item_id)
        if not metadata_path.is_file():
            missing.append(f"{item_id}:metadata")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["source_signature"] = signature
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        refreshed += 1
        if not feature_bundle_is_cached(
            store,
            item_id,
            tokens=True,
            action_latents=True,
            kinematic_latents=True,
            extractor_name="lawam_lam",
            source_signature=signature,
        ):
            missing.append(item_id)
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(
            f"Remote feature cache is incomplete for {len(missing)} records: {preview}"
        )
    if missing_sources:
        preview = ", ".join(missing_sources[:10])
        raise RuntimeError(
            f"Remote source artifacts are missing for {len(missing_sources)} paths: {preview}"
        )
    return {
        "candidates": len(examples),
        "references": len(refs),
        "signatures_refreshed": refreshed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a copied cache for remote ablations.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--success-refs", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--new-root", required=True)
    args = parser.parse_args()

    manifest_changes = rebase_jsonl(args.manifest, args.old_root, args.new_root)
    reference_changes = rebase_jsonl(args.success_refs, args.old_root, args.new_root)
    report = refresh_feature_signatures(args.manifest, args.success_refs, args.feature_root)
    report.update({
        "manifest_records_rebased": manifest_changes,
        "reference_records_rebased": reference_changes,
        "status": "ready",
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
