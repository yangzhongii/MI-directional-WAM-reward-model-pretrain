"""Convert official Robometer processed trajectories into base records."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from mi_reward.data.base_trajectory_schema import BaseTrajectory
from mi_reward.data.schema import SuccessReference, write_jsonl


@dataclass(frozen=True)
class TaskRule:
    pattern: re.Pattern[str]
    task_family: str
    physical_task_config: str
    object_prompts: dict[str, str]


def _load_rules(items: Any) -> list[TaskRule]:
    if not isinstance(items, list) or not items:
        raise ValueError("Robometer ingestion requires a non-empty task_rules list.")
    rules: list[TaskRule] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"task_rules[{index}] must be a mapping.")
        prompts = item.get("object_prompts")
        if not isinstance(prompts, dict) or not prompts:
            raise ValueError(f"task_rules[{index}].object_prompts must be a non-empty mapping.")
        rules.append(
            TaskRule(
                pattern=re.compile(str(item["match"])),
                task_family=str(item["task_family"]),
                physical_task_config=str(item["physical_task_config"]),
                object_prompts={str(key): str(value) for key, value in prompts.items()},
            )
        )
    return rules


def _match_rule(task: str, rules: Iterable[TaskRule]) -> TaskRule | None:
    return next((rule for rule in rules if rule.pattern.search(task)), None)


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "trajectory"


def _image_array(value: Any) -> np.ndarray | None:
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict) and value.get("bytes") is not None:
        from io import BytesIO

        try:
            return np.asarray(Image.open(BytesIO(value["bytes"])).convert("RGB"))
        except (OSError, ValueError):
            return None
    return None


def _existing_path(value: str | Path, dataset_root: Path) -> Path:
    """Resolve a cache path even when it was written on another machine."""

    path = Path(str(value)).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(dataset_root / path)
    else:
        candidates.append(dataset_root / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # Official processed caches sometimes preserve the source machine's
    # absolute path. Fall back to the unique basename under this dataset.
    matches = list(dataset_root.rglob(path.name))
    if len(matches) == 1 and matches[0].is_file():
        return matches[0].resolve()
    raise FileNotFoundError(f"Robometer media path is missing: {value}")


def _video_frames(path: Path) -> np.ndarray:
    """Decode a video file without depending on Robometer's optional decord API."""

    try:
        import imageio.v3 as imageio

        frames = np.asarray(imageio.imread(path, index=None))
    except Exception as exc:  # pragma: no cover - backend-specific decoder errors
        raise ValueError(f"Could not decode Robometer video: {path}") from exc
    if frames.ndim != 4:
        raise ValueError(f"Robometer video must decode to [T,H,W,C], got {frames.shape}: {path}")
    return frames


def _npz_frames(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "frames" in archive.files:
            frames = archive["frames"]
        elif archive.files:
            frames = archive[archive.files[0]]
        else:
            raise ValueError(f"Robometer NPZ contains no arrays: {path}")
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"Robometer NPZ frames must be [T,H,W,C], got {frames.shape}: {path}")
    return frames


def _materialize_item(item: Any, dataset_root: Path) -> list[Any]:
    """Expand one Robometer row value into image-like frame items."""

    if isinstance(item, dict) and item.get("path"):
        item = item["path"]
    if isinstance(item, (str, Path)):
        path = _existing_path(item, dataset_root)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            return [frame for frame in _npz_frames(path)]
        if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            return [frame for frame in _video_frames(path)]
        return [path]
    if isinstance(item, dict) and item.get("bytes") is not None:
        raw = item["bytes"]
        array = _image_array(item)
        if array is not None:
            return [array]
        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            handle.write(raw)
            handle.flush()
            return [frame for frame in _video_frames(Path(handle.name))]
    return [item]


def _materialize_frames(value: Any, output_dir: Path, dataset_root: Path) -> list[str]:
    if isinstance(value, (str, Path)) or isinstance(value, dict):
        items = _materialize_item(value, dataset_root)
    elif isinstance(value, np.ndarray) and value.ndim == 4:
        items = [value[index] for index in range(value.shape[0])]
    elif isinstance(value, (list, tuple)):
        items = []
        for item in value:
            items.extend(_materialize_item(item, dataset_root))
    else:
        raise ValueError(f"Unsupported Robometer frames value: {type(value).__name__}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, item in enumerate(items):
        array = _image_array(item)
        if array is None and isinstance(item, (str, Path)):
            path = _existing_path(item, dataset_root)
            array = np.asarray(Image.open(path).convert("RGB"))
        if array is None:
            raise ValueError(f"Unsupported Robometer frame item: {type(item).__name__}")
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3:
            raise ValueError(f"Robometer frame {index} must be HWC, got shape {array.shape}.")
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        elif array.shape[-1] >= 3:
            array = array[..., :3]
        if array.dtype != np.uint8:
            scale = 255.0 if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0 else 1.0
            array = np.clip(np.nan_to_num(array) * scale, 0, 255).astype(np.uint8)
        path = output_dir / f"{index:06d}.png"
        Image.fromarray(array).convert("RGB").save(path)
        paths.append(str(path.resolve()))
    if len(paths) < 2:
        raise ValueError("A Robometer base trajectory must contain at least two frames.")
    return paths


def _load_processed_dataset(processed_root: Path, dataset_name: str):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("Install the shared .venv before ingesting Robometer data.") from exc
    safe_name = dataset_name.replace("/", "_").replace(":", "_")
    candidates = [
        processed_root / dataset_name / "processed_dataset",
        processed_root / safe_name / "processed_dataset",
        processed_root / dataset_name,
        processed_root / safe_name,
    ]
    path = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if path is None:
        raise FileNotFoundError(
            f"Robometer processed dataset is missing under {processed_root}: {dataset_name!r}. "
            "Download and extract the Robometer processed cache first."
        )
    dataset_root = path.parent if path.name == "processed_dataset" else path
    return Dataset.load_from_disk(str(path), keep_in_memory=False), dataset_root


def ingest_robometer(
    config: dict[str, Any],
    output_records: str | Path,
    success_refs_path: str | Path,
) -> dict[str, Any]:
    """Write mapped Robometer trajectories and successful references.

    Robometer does not guarantee action, joint-state, calibration, or object
    pose sidecars. Those fields deliberately remain absent from base records.
    """

    processed_root = Path(str(config["processed_root"])).resolve()
    dataset_names = config.get("datasets")
    if not isinstance(dataset_names, list) or not dataset_names:
        raise ValueError("base_data.robometer.datasets must be a non-empty list.")
    rules = _load_rules(config.get("task_rules"))
    max_per_dataset = int(config.get("max_trajectories_per_dataset", -1))
    success_labels = {str(value).lower() for value in config.get("success_labels", ["successful", "success"])}
    output_path = Path(output_records).resolve()
    frame_root = output_path.parent / "robometer_frames"

    pending: list[dict[str, Any]] = []
    unmatched = 0
    for dataset_name_value in dataset_names:
        dataset_name = str(dataset_name_value)
        dataset, dataset_root = _load_processed_dataset(processed_root, dataset_name)
        limit = len(dataset) if max_per_dataset < 0 else min(len(dataset), max_per_dataset)
        for index in range(limit):
            row = dataset[index]
            if not isinstance(row, dict):
                continue
            task = str(row.get("task") or "").strip()
            rule = _match_rule(task, rules)
            if rule is None:
                unmatched += 1
                continue
            source_id = str(row.get("id") or f"{dataset_name}_{index:08d}")
            base_id = f"robometer/{_safe_id(dataset_name)}/{_safe_id(source_id)}"
            frame_value = row.get("frames")
            if frame_value is None:
                frame_value = row.get("frames_path", row.get("video"))
            frames = _materialize_frames(
                frame_value,
                frame_root / _safe_id(dataset_name) / _safe_id(source_id),
                dataset_root,
            )
            quality = str(row.get("quality_label") or "").lower()
            pending.append(
                {
                    "base_id": base_id,
                    "dataset_name": dataset_name,
                    "source_id": source_id,
                    "task": task,
                    "rule": rule,
                    "frames": frames,
                    "quality": quality,
                    "row": row,
                }
            )

    successful_by_task: dict[tuple[str, str], dict[str, Any]] = {}
    for item in pending:
        key = (item["rule"].task_family, item["task"])
        if item["quality"] in success_labels and key not in successful_by_task:
            successful_by_task[key] = item

    references: list[SuccessReference] = []
    reference_ids: dict[tuple[str, str], str] = {}
    for key, item in successful_by_task.items():
        ref_id = f"{item['base_id']}/success"
        reference_ids[key] = ref_id
        references.append(SuccessReference(ref_id=ref_id, task=item["task"], frames=item["frames"]))

    records: list[BaseTrajectory] = []
    skipped_without_reference = 0
    for item in pending:
        rule: TaskRule = item["rule"]
        key = (rule.task_family, item["task"])
        ref = successful_by_task.get(key)
        if ref is None:
            skipped_without_reference += 1
            continue
        row = item["row"]
        metadata = {
            "robometer_dataset": item["dataset_name"],
            "robometer_id": item["source_id"],
            "data_source": row.get("data_source"),
            "quality_label": row.get("quality_label"),
            "is_robot": row.get("is_robot"),
            "target_progress": row.get("target_progress"),
            "physical_sidecars_from_source": False,
        }
        records.append(
            BaseTrajectory(
                base_id=item["base_id"],
                source="robometer",
                task=item["task"],
                task_family=rule.task_family,
                instruction=item["task"],
                frames=item["frames"],
                initial_frame=item["frames"][0],
                goal_frame=ref["frames"][-1],
                goal_ref_id=reference_ids[key],
                physical_task_config=str(Path(rule.physical_task_config).resolve()),
                object_prompts=rule.object_prompts,
                split=str(config.get("split", "train")),
                metadata=metadata,
            )
        )

    if not records:
        raise ValueError(
            "Robometer ingestion produced no usable base trajectories. Check task_rules, success labels, and dataset names."
        )
    write_jsonl(output_path, records)
    write_jsonl(success_refs_path, references)
    report = {
        "source": "robometer",
        "output_records": str(output_path),
        "success_refs": str(Path(success_refs_path).resolve()),
        "records": len(records),
        "references": len(references),
        "unmatched": unmatched,
        "skipped_without_reference": skipped_without_reference,
        "datasets": [str(value) for value in dataset_names],
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Robometer processed data into base trajectory records.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--success-refs", required=True)
    args = parser.parse_args()
    payload = _load_yaml(Path(args.config))
    config = payload.get("robometer", payload)
    if not isinstance(config, dict):
        raise ValueError("Robometer config must be a mapping.")
    print(json.dumps(ingest_robometer(config, args.output, args.success_refs), indent=2))


if __name__ == "__main__":
    main()
