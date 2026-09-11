"""Adapt native LIBERO demonstrations or labeled dual-view trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

SUITES = ("libero_10", "libero_90", "libero_spatial", "libero_object", "libero_goal")


def build_qwen_row(language, agentview, wrist, indices, trajectory_id):
    """Preserve the shared builder's view-major, chronological history order."""
    if not isinstance(language, str) or not language.strip():
        raise ValueError("Trajectory requires task language")
    if not indices or len(agentview) != len(indices) or len(wrist) != len(indices):
        raise ValueError("Expected synchronized, nonempty dual-view history")
    images = [str(Path(p).resolve()) for p in [*agentview, *wrist]]
    for path in images:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return {
        "messages": [{"role": "user", "content":
                      f"Task: {language.rstrip('. ')}. Judge recent task progress as Positive, Unclear, or Negative."}],
        "images": images,
        "provenance": {
            "source": "libero",
            "trajectory_id": trajectory_id,
            "history_frame_indices": list(indices),
        },
    }


def iter_hdf5(data_root, benchmark, image_root, max_tasks=1, max_demos=1,
              history_frames=5, prefix_smoke=False):
    """Read only needed RGB slices; no simulator or privileged state is used.

    Native official demos are successful demonstrations. Their labels are not
    inferred from dones (which can also indicate timeouts in rollout datasets).
    """
    if benchmark not in SUITES:
        raise ValueError(benchmark)
    suite_dir = Path(data_root) / benchmark
    paths = sorted(suite_dir.glob("*_demo.hdf5"))
    if not paths:
        raise FileNotFoundError(f"No official LIBERO HDF5 demos in {suite_dir}")
    if history_frames < 1 or max_tasks < 0 or max_demos < 0:
        raise ValueError("history_frames must be positive; limits must be nonnegative")
    for path in paths[:max_tasks or None]:
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            info = json.loads(data.attrs["problem_info"])
            language = info["language_instruction"]
            task = path.stem.removesuffix("_demo")
            names = sorted(data.keys(), key=lambda name: int(name.removeprefix("demo_")))
            for name in names[:max_demos or None]:
                obs = data[name]["obs"]
                main, wrist = obs["agentview_rgb"], obs["eye_in_hand_rgb"]
                if main.shape != wrist.shape or len(main) < history_frames:
                    raise ValueError(f"Invalid synchronized history in {path}:{name}")
                if main.dtype != np.uint8 or main.ndim != 4 or main.shape[-1] != 3:
                    raise ValueError(f"Expected uint8 RGB [T,H,W,3] in {path}:{name}")
                trajectory_id = f"{benchmark}/{task}/{name}"
                endpoints = [("terminal", len(main))]
                if prefix_smoke and len(main) > 2 * history_frames:
                    endpoints.append(("early_prefix", max(history_frames, len(main) // 4)))
                for window, end in endpoints:
                    indices = list(range(end - history_frames, end))
                    folder = Path(image_root) / task / name
                    folder.mkdir(parents=True, exist_ok=True)
                    views = []
                    for view_name, frames in (("agentview", main), ("wrist", wrist)):
                        files = []
                        for index in indices:
                            target = folder / f"{view_name}_{index:06d}.png"
                            # Match the existing native demo reader/exporter orientation.
                            Image.fromarray(frames[index]).save(target)
                            files.append(target)
                        views.append(files)
                    yield {
                        "trajectory_id": trajectory_id, "task_id": task,
                        "window": window, "success": True if window == "terminal" else None,
                        "quality": None,
                        "label_source": "official_successful_demonstration" if window == "terminal" else None,
                        "source_path": str(path.resolve()), "num_frames": len(main),
                        "row": build_qwen_row(language, *views, indices, trajectory_id),
                    }


def iter_manifest(manifest, benchmark, history_frames=5, windows_per_trajectory=1):
    """Load real rollouts: ordered agentview/wrist paths, success, optional quality.

    Image paths are relative to the manifest. Higher numeric quality is better.
    Labels remain outside the model row. No generated/worldsample source allowed.
    """
    if history_frames < 1 or windows_per_trajectory < 1:
        raise ValueError("history_frames and windows_per_trajectory must be positive")
    manifest = Path(manifest).resolve()
    seen = set()
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["benchmark"] != benchmark:
            raise ValueError("Manifest benchmark mismatch")
        if record.get("source") != "libero_rollout":
            raise ValueError("Manifest requires source=libero_rollout")
        identity = (record["task_id"], record["trajectory_id"])
        if identity in seen:
            raise ValueError(f"Duplicate trajectory: {identity}")
        seen.add(identity)
        success, quality = record.get("success"), record.get("quality")
        if success is not None and type(success) is not bool:
            raise ValueError("success must be boolean or null")
        if quality is not None and (type(quality) not in (int, float) or not np.isfinite(quality)):
            raise ValueError("quality must be a finite number or null")
        main, wrist = record["agentview"], record["wrist"]
        if not isinstance(main, list) or not isinstance(wrist, list) or len(main) != len(wrist) or not main:
            raise ValueError("Manifest requires synchronized agentview/wrist path lists")
        ends = [len(main)] if windows_per_trajectory == 1 else sorted(set(
            np.linspace(min(history_frames, len(main)), len(main), windows_per_trajectory).round().astype(int).tolist()))
        for end in ends:
            indices = list(range(max(0, end - history_frames), end))
            views = [[manifest.parent / paths[i] for i in indices] for paths in (main, wrist)]
            if any("worldsample" in path.resolve().parts for view in views for path in view):
                raise ValueError("Worldsample is outside native LIBERO validation")
            yield {
                "trajectory_id": record["trajectory_id"], "task_id": record["task_id"],
                "policy_id": record.get("policy_id"), "pair_id": record.get("pair_id"),
                "window": "terminal" if end == len(main) else "trajectory_window",
                "success": success, "quality": quality,
                "label_source": "rollout_manifest", "source_path": str(manifest),
                "num_frames": len(main), "window_end": end,
                "row": build_qwen_row(record["language"], *views, indices, record["trajectory_id"]),
            }
