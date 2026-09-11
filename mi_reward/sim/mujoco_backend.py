"""Optional headless MuJoCo backend for the rigid-task MVP."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mi_reward.sim.base import InstanceAsset, SimulationRollout, SimulatorBackend, SimulatorConfig


class MuJoCoBackend(SimulatorBackend):
    """Thin MuJoCo adapter with no import-time MuJoCo dependency.

    Complex mesh replacement uses an asset-specific MuJoCo model path. Primitive
    variants may instead change the configured body mass and geom size in place.
    This keeps PushT/PushB and peg assets explicit rather than attempting an
    unsafe generic mesh mutation.
    """

    def __init__(self, config: SimulatorConfig):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is required for MuJoCoBackend. Install it in the reward environment with `pip install mujoco`.") from exc
        self._mj = mujoco
        self.config = config
        self._model_path = Path(config.model_path).resolve()
        self._model = None
        self._data = None
        self.reset()

    @property
    def model_path(self) -> Path:
        return self._model_path

    def _load_model(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo model not found: {path}")
        self._model_path = path.resolve()
        self._model = self._mj.MjModel.from_xml_path(str(self._model_path))
        self._data = self._mj.MjData(self._model)

    def reset(self, seed: int | None = None) -> None:
        del seed  # MuJoCo model initialization is deterministic for a fixed XML.
        self._load_model(self._model_path)
        self._mj.mj_resetData(self._model, self._data)
        self._mj.mj_forward(self._model, self._data)

    def apply_instance_variant(self, asset: InstanceAsset) -> None:
        if asset.model_path is not None:
            self._load_model(Path(asset.model_path))
            return
        if self.config.object_body_name and asset.mass is not None:
            body_id = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_BODY, self.config.object_body_name)
            if body_id < 0:
                raise ValueError(f"Configured object body does not exist: {self.config.object_body_name}")
            self._model.body_mass[body_id] = float(asset.mass)
        if self.config.object_geom_name and asset.geom_size is not None:
            geom_id = self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_GEOM, self.config.object_geom_name)
            if geom_id < 0:
                raise ValueError(f"Configured object geom does not exist: {self.config.object_geom_name}")
            if len(asset.geom_size) > self._model.geom_size.shape[1]:
                raise ValueError("InstanceAsset.geom_size has more values than MuJoCo geom_size supports.")
            self._model.geom_size[geom_id, : len(asset.geom_size)] = asset.geom_size
        self._mj.mj_forward(self._model, self._data)

    def rollout(self, actions: np.ndarray) -> SimulationRollout:
        if actions.ndim != 2:
            raise ValueError(f"Expected actions [T, A], got {tuple(actions.shape)}")
        if actions.shape[1] > self._model.nu:
            raise ValueError(f"Action dimension {actions.shape[1]} exceeds MuJoCo actuator count {self._model.nu}.")
        qpos, qvel = [], []
        for action in actions:
            self._data.ctrl[:] = 0.0
            self._data.ctrl[: action.shape[0]] = action
            self._mj.mj_step(self._model, self._data)
            qpos.append(np.array(self._data.qpos, copy=True))
            qvel.append(np.array(self._data.qvel, copy=True))
        return SimulationRollout(qpos=np.stack(qpos), qvel=np.stack(qvel), actions=np.array(actions, copy=True))

    def render_rgb(self, width: int, height: int) -> np.ndarray:
        renderer = self._mj.Renderer(self._model, height=height, width=width)
        try:
            renderer.update_scene(self._data, camera=self.config.camera_name)
            return renderer.render()
        finally:
            renderer.close()
