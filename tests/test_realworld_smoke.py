"""Smoke tests for deployment/realworld pure utility modules.

These tests run without ROS, Franka hardware, or a policy server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

# Ensure the repo root is on sys.path so that deployment.realworld is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# FrankaRobotState
# ---------------------------------------------------------------------------

class TestFrankaRobotState:
    def test_default_construction(self):
        from deployment.realworld.franka_robot_state import FrankaRobotState

        state = FrankaRobotState()
        assert state.tcp_pose.shape == (7,)
        assert state.tcp_vel.shape == (6,)
        assert state.arm_joint_position.shape == (7,)
        assert state.arm_joint_velocity.shape == (7,)
        assert state.tcp_force.shape == (3,)
        assert state.tcp_torque.shape == (3,)
        assert state.arm_jacobian.shape == (6, 7)
        assert state.gripper_position == 0
        assert state.gripper_open is False
        assert state.hand_position is None

    def test_partial_construction(self):
        from deployment.realworld.franka_robot_state import FrankaRobotState

        state = FrankaRobotState(
            tcp_pose=np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
            gripper_position=100,
            gripper_open=True,
        )
        assert np.allclose(state.tcp_pose[:3], [0.5, 0.0, 0.3])
        assert state.gripper_position == 100
        assert state.gripper_open is True

    def test_to_dict(self):
        from deployment.realworld.franka_robot_state import FrankaRobotState

        state = FrankaRobotState(tcp_pose=np.ones(7))
        d = state.to_dict()
        assert isinstance(d, dict)
        assert "tcp_pose" in d
        assert "tcp_vel" in d
        assert "gripper_open" in d


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

class TestUtils:
    def test_normalize_unit_quaternion(self):
        from deployment.realworld.utils import normalize

        q = np.array([0.0, 0.0, 0.0, 1.0])
        result = normalize(q)
        assert np.allclose(result, q)

    def test_normalize_scaled_quaternion(self):
        from deployment.realworld.utils import normalize

        q = np.array([0.0, 0.0, 0.0, 2.0])
        result = normalize(q)
        assert np.allclose(np.linalg.norm(result), 1.0)
        assert np.allclose(result, [0.0, 0.0, 0.0, 1.0])

    def test_normalize_zero_raises(self):
        from deployment.realworld.utils import normalize

        with pytest.raises(ValueError, match="Zero-norm"):
            normalize(np.zeros(4))

    def test_wrap_to_pi(self):
        from deployment.realworld.utils import wrap_to_pi

        assert wrap_to_pi(0.0) == pytest.approx(0.0)
        assert wrap_to_pi(np.pi) == pytest.approx(-np.pi)  # interval is [-pi, pi)
        assert wrap_to_pi(1.5 * np.pi) == pytest.approx(-0.5 * np.pi)
        assert wrap_to_pi(-1.5 * np.pi) == pytest.approx(0.5 * np.pi)
        arr = np.array([0.0, 1.5 * np.pi, -1.5 * np.pi])
        result = wrap_to_pi(arr)
        assert result[0] == pytest.approx(0.0)

    def test_quat_slerp_identity(self):
        from deployment.realworld.utils import quat_slerp

        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = np.array([0.0, 0.0, 0.0, 1.0])
        result = quat_slerp(q0, q1, 5)
        assert result.shape == (5, 4)
        for q in result:
            assert np.allclose(q, q0)

    def test_quat_slerp_shortest_path(self):
        from deployment.realworld.utils import quat_slerp

        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = np.array([0.0, 0.0, 0.0, -1.0])  # same rotation
        result = quat_slerp(q0, q1, 3)
        # Should all be identity quaternion (took shortest path)
        for q in result:
            assert np.allclose(np.abs(q[3]), 1.0, atol=1e-6)

    def test_quat_slerp_interpolation(self):
        from deployment.realworld.utils import quat_slerp

        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = R.from_euler("z", np.pi / 2).as_quat()
        result = quat_slerp(q0, q1, np.array([0.0, 0.5, 1.0]))
        assert result.shape == (3, 4)
        assert np.allclose(result[0], q0)
        assert np.allclose(result[-1], q1)

    def test_clip_euler_no_clip_needed(self):
        from deployment.realworld.utils import clip_euler_to_target_window

        euler = np.array([0.1, 0.2, 0.3])
        target = np.array([0.1, 0.2, 0.3])
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        result = clip_euler_to_target_window(euler, target, lower, upper)
        assert np.allclose(result, euler)

    def test_clip_euler_respects_lower_bound(self):
        from deployment.realworld.utils import clip_euler_to_target_window

        euler = np.array([-2.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, 0.0])
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        result = clip_euler_to_target_window(euler, target, lower, upper)
        assert result[0] >= -1.0

    def test_construct_adjoint_matrix(self):
        from deployment.realworld.utils import construct_adjoint_matrix

        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        adj = construct_adjoint_matrix(pose)
        assert adj.shape == (6, 6)
        # Top-left 3x3 = rotation (identity here)
        assert np.allclose(adj[:3, :3], np.eye(3))
        # Bottom-right 3x3 = rotation (identity here)
        assert np.allclose(adj[3:, 3:], np.eye(3))

    def test_construct_homogeneous_matrix(self):
        from deployment.realworld.utils import construct_homogeneous_matrix

        pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
        T = construct_homogeneous_matrix(pose)
        assert T.shape == (4, 4)
        assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
        assert T[3, 3] == 1.0
        assert T[3, 0] == 0.0


# ---------------------------------------------------------------------------
# EndEffectorType
# ---------------------------------------------------------------------------

class TestEndEffectorType:
    def test_enum_values(self):
        from deployment.realworld.end_effectors.base import EndEffectorType

        assert EndEffectorType.FRANKA_GRIPPER.value == "franka_gripper"
        assert EndEffectorType.ROBOTIQ_GRIPPER.value == "robotiq_gripper"
        assert EndEffectorType.RUIYAN_HAND.value == "ruiyan_hand"

    def test_is_gripper(self):
        from deployment.realworld.end_effectors.base import EndEffectorType

        assert EndEffectorType.FRANKA_GRIPPER.is_gripper is True
        assert EndEffectorType.ROBOTIQ_GRIPPER.is_gripper is True
        assert EndEffectorType.RUIYAN_HAND.is_gripper is False

    def test_is_hand(self):
        from deployment.realworld.end_effectors.base import EndEffectorType

        assert EndEffectorType.RUIYAN_HAND.is_hand is True
        assert EndEffectorType.FRANKA_GRIPPER.is_hand is False

    def test_gripper_backend(self):
        from deployment.realworld.end_effectors.base import EndEffectorType

        assert EndEffectorType.FRANKA_GRIPPER.gripper_backend == "franka"
        assert EndEffectorType.ROBOTIQ_GRIPPER.gripper_backend == "robotiq"
        with pytest.raises(ValueError):
            EndEffectorType.RUIYAN_HAND.gripper_backend

    def test_normalize_end_effector_type(self):
        from deployment.realworld.end_effectors.base import (
            EndEffectorType,
            normalize_end_effector_type,
        )

        assert normalize_end_effector_type("franka_gripper") == EndEffectorType.FRANKA_GRIPPER
        assert normalize_end_effector_type("ruiyan_hand") == EndEffectorType.RUIYAN_HAND
        assert normalize_end_effector_type("franka_gripper", "robotiq") == EndEffectorType.ROBOTIQ_GRIPPER


# ---------------------------------------------------------------------------
# Imports (module-level, no instantiation)
# ---------------------------------------------------------------------------

def test_can_import_pure_modules():
    """All pure (non-ROS) modules must be importable."""
    import deployment.realworld.franka_robot_state  # noqa: F811
    import deployment.realworld.utils  # noqa: F811
    import deployment.realworld.end_effectors.base  # noqa: F811
    import deployment.realworld.end_effectors  # noqa: F811
