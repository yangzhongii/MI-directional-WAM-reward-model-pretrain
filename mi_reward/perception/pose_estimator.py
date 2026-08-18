"""Pose adapter for calibrated sensors, markers, or simulator state."""

from __future__ import annotations

from pathlib import Path

from mi_reward.data.instance_schema import ObjectStateSequence, load_object_state_sequence


class SidecarPoseEstimator:
    """Use a recorded or simulator object-state sidecar as the pose authority."""

    def estimate(self, object_state_path: str | Path) -> ObjectStateSequence:
        return load_object_state_sequence(object_state_path)
