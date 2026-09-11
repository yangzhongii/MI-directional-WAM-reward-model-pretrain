"""Snapshot frozen v3 artifacts and verify their hashes without altering sources."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path("logs/mi_reward/v3_teacher_mainline")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = Path(args.output_dir).resolve()
    if args.verify:
        frozen = json.loads((output / "freeze_manifest.json").read_text())
        errors = []
        for record in frozen["files"]:
            source = root / record["source"]
            if not source.is_file() or digest(source) != record["sha256"]:
                errors.append(record["source"])
            if record.get("snapshot"):
                snapshot = output / record["snapshot"]
                if not snapshot.is_file() or digest(snapshot) != record["sha256"]:
                    errors.append(record["snapshot"])
        result = {"verified_at": datetime.now(timezone.utc).isoformat(), "files": len(frozen["files"]),
                  "unchanged": not errors, "mismatches": errors}
        (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if errors:
            raise RuntimeError("Frozen baseline changed")
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / "freeze_manifest.json").exists() or (output / "snapshot").exists():
        raise FileExistsError("Choose a fresh freeze directory")
    sources = [Path("MI_Directional_Potential_Research_Idea_v3.md"),
               BASE / "teacher_smoke5", BASE / "teacher_export5_v2", BASE / "features",
               BASE / "qwen3_vl_reward_v1/model", BASE / "qwen3_vl_reward_v1/run_config.json",
               BASE / "qwen3_vl_reward_v1/eval_full536.json",
               BASE / "qwen3_vl_reward_v1/eval_full536.predictions.jsonl",
               Path("mi_reward/scoring/information_teacher_v3.py"),
               Path("mi_reward/scoring/teacher_consistency_v3.py"),
               Path("mi_reward/training/train_information_teacher_v3.py"),
               Path("mi_reward/training/qwen3_vl_reward_data_v3.py"),
               Path("mi_reward/data/export_teacher_targets_v3.py"),
               Path("mi_reward/data/libero_privileged.py"),
               Path("mi_reward/features/lawam_lam_extractor.py"),
               Path("mi_reward/inference/qwen3_vl_reward.py"),
               Path("latent_action_model/core"),
               Path(".venv/models/lawam_lam/dino_large_vae.yaml")]
    dependencies = [Path(".venv/models/lawam_lam/checkpoints/pytorch_model.pt"),
                    Path(".venv/models/dinov3-vitb16-pretrain-lvd1689m"),
                    BASE / "validation_audit_v1/rollouts", BASE / "validation_audit_v1/validation"]
    records = []
    for snapshot, paths in ((True, sources), (False, dependencies)):
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
            files = sorted(path.rglob("*")) if path.is_dir() else [path]
            for source in files:
                if not source.is_file() or "__pycache__" in source.parts:
                    continue
                record = {"source": str(source), "size": source.stat().st_size, "sha256": digest(source)}
                if snapshot:
                    target = output / "snapshot" / source
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    if digest(target) != record["sha256"]:
                        raise RuntimeError(f"Snapshot mismatch: {source}")
                    target.chmod(0o444)
                    record["snapshot"] = str(target.relative_to(output))
                records.append(record)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "workspace": str(root),
                "policy": "No formula, calibration, teacher/Qwen weights or training changes during audit.",
                "gamma": 0.99, "beta": 1.0, "qwen_history_frames": 5,
                "git": "Workspace .git metadata unavailable; SHA256 and file snapshots are the baseline.",
                "files": records}
    (output / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "freeze_manifest.json").chmod(0o444)
    print(json.dumps({"freeze_manifest": str(output / "freeze_manifest.json"), "files": len(records),
                      "snapshot_files": sum("snapshot" in r for r in records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
