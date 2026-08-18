"""Write synchronized rollout sidecars from a simulator or test fixture."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from mi_reward.data.instance_schema import ObjectStateFrame
from mi_reward.data.schema import ControlArtifacts, SimulationProvenance
from mi_reward.relations.sequence import RelationSequence
from mi_reward.sim.base import SimulationRollout


@dataclass(frozen=True)
class RolloutArtifacts:
    frame_paths: list[str]
    action_path: str
    robot_state_path: str
    object_state_path: str
    relation_path: str
    controls: ControlArtifacts
    simulation_path: str


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record) if hasattr(record, "__dataclass_fields__") else record
            handle.write(json.dumps(payload) + "\n")


def write_rollout_artifacts(
    output_dir: str | Path,
    *,
    frames: Iterable[np.ndarray],
    rollout: SimulationRollout,
    robot_states: list[dict[str, object]],
    object_states: list[ObjectStateFrame],
    relations: RelationSequence,
    simulation: SimulationProvenance,
    masks: Iterable[np.ndarray] | None = None,
    depths: Iterable[np.ndarray] | None = None,
) -> RolloutArtifacts:
    """Persist one synchronized candidate without invoking any model worker."""

    root = Path(output_dir)
    frame_dir, mask_dir, depth_dir = root / "frames", root / "masks", root / "depth"
    frame_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    frame_items = list(frames)
    # NumPy arrays deliberately remain valid per-frame iterables.  Using
    # ``masks or []`` here would try to coerce an array to bool and fail.
    mask_items = list(masks) if masks is not None else []
    depth_items = list(depths) if depths is not None else []
    steps = len(frame_items)
    if not (steps == len(rollout.actions) == len(robot_states) == len(object_states) == len(relations.values)):
        raise ValueError("Rollout frames, actions, states, objects, and relations must share a timestep count.")
    if mask_items and len(mask_items) != steps:
        raise ValueError("Mask count must match frame count.")
    if depth_items and len(depth_items) != steps:
        raise ValueError("Depth count must match frame count.")
    frame_paths: list[str] = []
    for index, frame in enumerate(frame_items):
        path = frame_dir / f"frame_{index:06d}.png"
        Image.fromarray(np.asarray(frame).astype(np.uint8)).save(path)
        frame_paths.append(str(path.resolve()))
    for index, mask in enumerate(mask_items):
        Image.fromarray(np.asarray(mask).astype(np.uint8)).save(mask_dir / f"frame_{index:06d}.png")
    for index, depth in enumerate(depth_items):
        np.save(depth_dir / f"frame_{index:06d}.npy", np.asarray(depth))
    action_path = root / "actions.npy"
    np.save(action_path, rollout.actions)
    robot_state_path = root / "robot_states.jsonl"
    _write_jsonl(robot_state_path, robot_states)
    object_state_path = root / "object_states.jsonl"
    _write_jsonl(object_state_path, object_states)
    relation_path = root / "relations.jsonl"
    _write_jsonl(
        relation_path,
        [{"names": relations.names, "values": values.tolist(), "source": "simulator"} for values in relations.values],
    )
    simulation_path = root / "simulation.json"
    simulation_path.write_text(json.dumps(asdict(simulation)), encoding="utf-8")
    return RolloutArtifacts(
        frame_paths=frame_paths,
        action_path=str(action_path.resolve()),
        robot_state_path=str(robot_state_path.resolve()),
        object_state_path=str(object_state_path.resolve()),
        relation_path=str(relation_path.resolve()),
        controls=ControlArtifacts(mask_root=str(mask_dir.resolve()), depth_root=str(depth_dir.resolve())),
        simulation_path=str(simulation_path.resolve()),
    )
