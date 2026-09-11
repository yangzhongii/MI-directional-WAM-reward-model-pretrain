from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from mi_reward.scripts.prepare_mujoco_assets import SCENE_VARIANTS, prepare_assets


MINIMAL_PANDA = """<mujoco model="panda">
  <compiler meshdir="assets"/>
  <worldbody>
    <body name="link0">
      <body name="hand">
        <body name="left_finger"><joint name="finger_joint1"/><geom class="collision" mesh="finger_0"/></body>
        <body name="right_finger"><joint name="finger_joint2"/><geom class="collision" mesh="finger_0"/></body>
      </body>
    </body>
  </worldbody>
  <tendon><fixed name="split"/></tendon>
  <equality><joint joint1="finger_joint1" joint2="finger_joint2"/></equality>
  <actuator>
    <general name="actuator1"/><general name="actuator2"/><general name="actuator3"/>
    <general name="actuator4"/><general name="actuator5"/><general name="actuator6"/>
    <general name="actuator7"/><general name="actuator8"/>
  </actuator>
  <keyframe><key name="home"/></keyframe>
  <contact><exclude body1="link0" body2="hand"/></contact>
</mujoco>
"""


def test_prepare_assets_adapts_panda_and_writes_all_scene_variants(tmp_path: Path) -> None:
    menagerie = tmp_path / "mujoco_menagerie" / "franka_emika_panda"
    (menagerie / "assets").mkdir(parents=True)
    (menagerie / "panda.xml").write_text(MINIMAL_PANDA, encoding="utf-8")
    (menagerie / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    output = tmp_path / "custom_task" / "scene"

    paths = prepare_assets(menagerie, output, validate=False)

    assert {path.name for path in paths} == set(SCENE_VARIANTS)
    adapted = ET.parse(output / "panda_menagerie_adapted.xml").getroot()
    meshdir = adapted.find("compiler").get("meshdir")
    assert meshdir is not None
    assert not Path(meshdir).is_absolute()
    assert (output / meshdir).resolve() == (menagerie / "assets").resolve()
    assert adapted.find(".//body[@name='panda']") is not None
    assert adapted.find(".//body[@name='left_finger']") is None
    assert adapted.find("tendon") is None
    assert adapted.find("equality") is None
    assert adapted.find("keyframe") is None
    assert adapted.find(".//general[@name='actuator8']") is None

    for path in paths:
        root = ET.parse(path).getroot()
        camera = root.find(".//camera[@name='global_cam']")
        assert camera is not None
        assert camera.get("pos") == "1.05 -0.90 1.45"
        assert camera.get("fovy") == "42"
        assert root.find(".//body[@name='panda_hand']") is not None
        assert root.find(".//body[@name='task_object']") is not None
        assert root.find(".//geom[@name='task_object_geom']") is not None
        gripper = root.find(".//position[@name='gripper']")
        assert gripper is not None
        assert gripper.get("ctrlrange") == "0 0.05"
        assert root.find(".//key[@name='home']") is not None
