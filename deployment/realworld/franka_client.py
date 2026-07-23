#!/usr/bin/env python3
"""Franka real-world client using RLinf's FCI controller + starVLA HTTP policy server.

Usage::

    python examples/Franka/franka_client.py \\
        --robot_ip 172.16.0.2 \\
        --policy_url http://<gpu_server>:9876 \\
        --instruction "peg insertion" \\
        --camera_ids 0 2
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from .franka_controller import FrankaController
from .franka_robot_state import FrankaRobotState
from .utils import quat_slerp

LOGGER = logging.getLogger(__name__)


def _encode_image_bgr(image: np.ndarray) -> str:
    """Encode a BGR image as a base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def _http_post(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class FrankaClient:
    """Main control loop for Franka real-world deployment."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._cameras: list[cv2.VideoCapture] = []
        self._controller: FrankaController | None = None
        self._step_count = 0
        self._recording: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Initialise controller and cameras."""
        LOGGER.info("Setting up Franka controller (robot_ip=%s)...", self.args.robot_ip)
        self._controller = FrankaController(
            robot_ip=self.args.robot_ip,
            ros_pkg=self.args.ros_pkg,
            end_effector_type=self.args.end_effector_type,
            gripper_type=self.args.gripper_type,
            gripper_connection=self.args.gripper_connection,
        )

        LOGGER.info("Waiting for robot to be ready...")
        start = time.time()
        while not self._controller.is_robot_up():
            if time.time() - start > 30:
                raise TimeoutError("Franka robot did not become ready within 30s.")
            time.sleep(0.5)
        LOGGER.info("Robot is ready.")

        # Open cameras
        for cam_id in self.args.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {cam_id}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.camera_height)
            cap.set(cv2.CAP_PROP_FPS, self.args.camera_fps)
            self._cameras.append(cap)
            LOGGER.info("Camera %d opened: %dx%d", cam_id, self.args.camera_width, self.args.camera_height)
        if not self._cameras:
            raise RuntimeError("At least one camera is required.")

        # Record directory
        self._record_dir = Path(self.args.record_dir).expanduser().resolve()
        self._record_dir.mkdir(parents=True, exist_ok=True)

    def teardown(self) -> None:
        """Release resources."""
        for cap in self._cameras:
            cap.release()
        self._cameras.clear()
        if self._controller is not None:
            self._controller.stop_impedance()
        LOGGER.info("Client shutdown complete.")

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _capture_frames(self) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for cap in self._cameras:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera read failed.")
            frames.append(frame)
        return frames

    # ------------------------------------------------------------------
    # Robot state
    # ------------------------------------------------------------------

    def _get_robot_state(self) -> dict[str, Any]:
        if self._controller is None:
            raise RuntimeError("Controller not initialised.")
        state: FrankaRobotState = self._controller.get_state()
        return {
            "ee_position": state.tcp_pose[:3].astype(np.float64).tolist(),
            "ee_quat_wxyz": np.roll(state.tcp_pose[3:], 1).astype(np.float64).tolist(),
            "gripper_width": float(state.gripper_position),
            "tcp_force": state.tcp_force.astype(np.float64).tolist(),
            "tcp_torque": state.tcp_torque.astype(np.float64).tolist(),
            "joint_position": state.arm_joint_position.astype(np.float64).tolist(),
        }

    # ------------------------------------------------------------------
    # Policy communication
    # ------------------------------------------------------------------

    def _query_policy(self) -> dict[str, Any]:
        """Send current observation to policy server and return response."""
        frames = self._capture_frames()
        robot_state = self._get_robot_state()

        payload: dict[str, Any] = {
            "primary_image": [_encode_image_bgr(f) for f in frames],
            "lang": self.args.instruction,
            "state": robot_state,
            "embodiment_id": int(self.args.embodiment_id),
            "action_hz": float(self.args.action_hz),
            "record_intermediates": self.args.record_intermediates,
        }

        resp = _http_post(f"{self.args.policy_url}/act", payload)
        if resp.get("status") != "ok":
            raise RuntimeError(f"Policy server error: {resp.get('message', resp)}")
        return resp

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action_chunk: np.ndarray, step_hz: float) -> None:
        """Execute a chunk of actions, one step at a time."""
        chunk_len = action_chunk.shape[0]
        step_dt = 1.0 / step_hz

        for i in range(chunk_len):
            action = action_chunk[i]  # [x_delta, y_delta, z_delta, rx_delta, ry_delta, rz_delta, gripper]
            xyz_delta = action[:3] * self.args.action_scale[0]
            rpy_delta = action[3:6] * self.args.action_scale[1]
            gripper_action = action[6] * self.args.action_scale[2]

            # Get current state
            if self._controller is None:
                raise RuntimeError("Controller not initialised.")
            state: FrankaRobotState = self._controller.get_state()
            current_xyz = state.tcp_pose[:3].copy()
            current_quat = state.tcp_pose[3:].copy()

            # Compute target pose
            target_xyz = current_xyz + xyz_delta
            delta_rot = R.from_euler("xyz", rpy_delta)
            current_rot = R.from_quat(current_quat)
            target_quat = (delta_rot * current_rot).as_quat()
            target_pose = np.concatenate([target_xyz, target_quat]).astype(np.float32)

            # Send to robot
            self._controller.move_arm(target_pose)
            self._controller.command_end_effector(np.array([gripper_action]))

            # Record
            if self.args.record:
                self._recording.append({
                    "step": self._step_count,
                    "action": action.tolist(),
                    "target_pose": target_pose.tolist(),
                    "gripper": float(gripper_action),
                    "tcp_pose": state.tcp_pose.tolist(),
                    "tcp_force": state.tcp_force.tolist(),
                })

            self._step_count += 1
            time.sleep(step_dt)

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def _save_recording(self) -> None:
        if not self._recording:
            return
        out_path = self._record_dir / f"recording_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        arrays = {k: np.array([r[k] for r in self._recording]) for k in self._recording[0]}
        np.savez_compressed(out_path, **arrays)
        LOGGER.info("Saved recording (%d steps) to %s", len(self._recording), out_path)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.setup()
        LOGGER.info(
            "Starting control loop (action_hz=%.1f, policy_url=%s, instruction=%s)",
            self.args.action_hz,
            self.args.policy_url,
            self.args.instruction,
        )
        try:
            while self._step_count < self.args.max_steps:
                loop_start = time.time()

                # 1. Query policy
                resp = self._query_policy()

                # 2. Execute action chunk
                actions = np.asarray(resp["actions"], dtype=np.float32)
                self._execute_action(actions, step_hz=self.args.action_hz)

                # 3. Log
                loop_time = time.time() - loop_start
                LOGGER.debug(
                    "Step %d/%d, chunk_len=%d, loop_time=%.3fs",
                    self._step_count,
                    self.args.max_steps,
                    actions.shape[0],
                    loop_time,
                )
        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user.")
        finally:
            self._save_recording()
            self.teardown()

        LOGGER.info("Done — %d steps executed.", self._step_count)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Franka real-world client with starVLA HTTP policy server.")

    # Robot
    p.add_argument("--robot_ip", type=str, required=True, help="Franka robot IP address.")
    p.add_argument("--ros_pkg", type=str, default="serl_franka_controllers")
    p.add_argument("--end_effector_type", type=str, default="franka_gripper")
    p.add_argument("--gripper_type", type=str, default="franka")
    p.add_argument("--gripper_connection", type=str, default=None)

    # Policy server
    p.add_argument("--policy_url", type=str, default="http://localhost:9876", help="starVLA HTTP policy server URL.")
    p.add_argument("--instruction", type=str, required=True, help="Task instruction string.")
    p.add_argument("--embodiment_id", type=int, default=25)
    p.add_argument("--action_hz", type=float, default=5.0)
    p.add_argument("--record_intermediates", action="store_true", help="Request LAM feature intermediates from server.")

    # Camera
    p.add_argument("--camera_ids", type=int, nargs="+", default=[0], help="OpenCV camera device IDs.")
    p.add_argument("--camera_width", type=int, default=640)
    p.add_argument("--camera_height", type=int, default=480)
    p.add_argument("--camera_fps", type=int, default=30)

    # Action
    p.add_argument("--action_scale", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                   help="Scale factors for [xyz_delta, rpy_delta, gripper].")

    # Control
    p.add_argument("--max_steps", type=int, default=1000, help="Maximum number of control steps.")

    # Recording
    p.add_argument("--record", action="store_true", help="Record all steps to a .npz file.")
    p.add_argument("--record_dir", type=str, default="logs/franka_client_recordings")

    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_argparser().parse_args()
    client = FrankaClient(args)
    client.run()


if __name__ == "__main__":
    main()
