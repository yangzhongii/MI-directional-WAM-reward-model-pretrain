#!/usr/bin/env python3
"""Build the rigid-task MuJoCo scenes from the open Franka Panda asset.

The robot meshes and MJCF come from MuJoCo Menagerie.  Task objects are
deliberately made from native MuJoCo primitives so the generated scenes stay
small, redistributable, and easy to vary without another large asset pack.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
OBJECT_SCENES = {
    "pick_apple": ("pick_place", "apple"),
    "banana": ("pick_place", "banana"),
    "push_t": ("push_shape", "push_t"),
    "push_b": ("push_shape", "push_b"),
    "peg_round": ("peg_insertion", "peg_round"),
    "peg_square": ("peg_insertion", "peg_square"),
}
SCENE_APPEARANCES = {
    "table_light_1": {
        "sky_rgb1": "0.65 0.76 0.90",
        "sky_rgb2": "0.12 0.16 0.22",
        "table_rgb1": "0.72 0.72 0.70",
        "table_rgb2": "0.58 0.58 0.56",
        "floor_rgba": "0.16 0.18 0.20 1",
        "wall_rgba": "0.76 0.78 0.80 1",
        "headlight_diffuse": "0.65 0.65 0.65",
        "headlight_ambient": "0.35 0.35 0.35",
        "light_pos": "0.2 -0.4 1.8",
    },
    "table_dark_2": {
        "sky_rgb1": "0.20 0.23 0.28",
        "sky_rgb2": "0.025 0.030 0.040",
        "table_rgb1": "0.20 0.22 0.24",
        "table_rgb2": "0.08 0.09 0.11",
        "floor_rgba": "0.055 0.060 0.070 1",
        "wall_rgba": "0.16 0.18 0.22 1",
        "headlight_diffuse": "0.42 0.40 0.36",
        "headlight_ambient": "0.12 0.13 0.16",
        "light_pos": "0.0 -0.2 1.55",
    },
}

# Public manifest used by tests and tooling.  The unsuffixed files remain as
# backwards-compatible aliases of table_light_1; the suffixed files are the
# native scene-level augmentation assets consumed by the planner.
SCENE_VARIANTS = {
    **{
        f"scene_{object_name}_{scene_id}.xml": (task_family, object_variant, scene_id)
        for object_name, (task_family, object_variant) in OBJECT_SCENES.items()
        for scene_id in SCENE_APPEARANCES
    },
    **{
        f"scene_{object_name}.xml": (task_family, object_variant, "table_light_1")
        for object_name, (task_family, object_variant) in OBJECT_SCENES.items()
    },
    "scene.xml": ("pick_place", "apple", "table_light_1"),
}


def _find_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    raise ValueError(f"Menagerie Panda MJCF is missing <{tag} name={name!r}>")


def _adapt_panda(source: Path, target: Path, mesh_dir: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"Menagerie Panda MJCF has no compiler section: {source}")
    # Keep generated MJCF relocatable across copied repositories, containers,
    # and rsync workers. Absolute controller paths break as soon as a remote
    # node uses a different repository root.
    compiler.set("meshdir", os.path.relpath(mesh_dir.resolve(), target.parent.resolve()))

    panda = _find_named(root, "body", "link0")
    panda.set("name", "panda")
    hand = _find_named(root, "body", "hand")

    for element in root.iter():
        for attribute in ("body", "body1", "body2"):
            if element.get(attribute) == "link0":
                element.set(attribute, "panda")
    # The Menagerie arm remains the high-quality visual robot.  Cartesian data
    # generation uses a separate mocap proxy gripper below; welding a mocap body
    # to Menagerie's high-gain joint servos creates an over-constrained system.
    for child in list(hand):
        if child.tag == "body" and child.get("name") in {"left_finger", "right_finger"}:
            hand.remove(child)

    tendon = root.find("tendon")
    if tendon is not None:
        root.remove(tendon)
    equality = root.find("equality")
    if equality is not None:
        root.remove(equality)
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError(f"Menagerie Panda MJCF has no actuator section: {source}")
    original_gripper = next(
        (item for item in actuator.findall("general") if item.get("name") == "actuator8"),
        None,
    )
    if original_gripper is None:
        raise ValueError("Menagerie Panda MJCF is missing actuator8")
    actuator.remove(original_gripper)

    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)

    ET.indent(tree, space="  ")
    target.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding="unicode", xml_declaration=False)
    with target.open("a", encoding="utf-8") as handle:
        handle.write("\n")


def _dummy_object_geoms(start: int, count: int = 5) -> str:
    return "\n".join(
        f'      <geom name="task_object_geom_aux{index}" type="sphere" size="0.001" '
        'rgba="0 0 0 0" contype="0" conaffinity="0" mass="0.000001"/>'
        for index in range(start, count)
    )


def _object_geoms(variant: str) -> str:
    common = 'friction="1.2 0.01 0.001" density="550"'
    if variant == "apple":
        real = [
            f'      <geom name="task_object_geom" type="ellipsoid" size="0.035 0.035 0.040" '
            f'rgba="0.78 0.06 0.04 1" {common}/>'
        ]
    elif variant == "banana":
        real = [
            f'      <geom name="task_object_geom" type="capsule" fromto="-0.050 0 0 0.005 0 0.012" '
            f'size="0.016" rgba="0.95 0.78 0.06 1" {common}/>',
            f'      <geom name="task_object_geom_aux1" type="capsule" fromto="0.005 0 0.012 0.055 0 0" '
            f'size="0.016" rgba="0.95 0.78 0.06 1" {common}/>',
        ]
    elif variant == "push_t":
        real = [
            f'      <geom name="task_object_geom" type="box" pos="0 -0.018 0" size="0.018 0.052 0.018" '
            f'rgba="0.12 0.48 0.82 1" {common}/>',
            f'      <geom name="task_object_geom_aux1" type="box" pos="0 0.040 0" size="0.060 0.018 0.018" '
            f'rgba="0.12 0.48 0.82 1" {common}/>',
        ]
    elif variant == "push_b":
        real = [
            f'      <geom name="task_object_geom" type="box" pos="-0.040 0 0" size="0.016 0.065 0.018" '
            f'rgba="0.68 0.20 0.74 1" {common}/>',
            f'      <geom name="task_object_geom_aux1" type="box" pos="0 0.049 0" size="0.040 0.016 0.018" '
            f'rgba="0.68 0.20 0.74 1" {common}/>',
            f'      <geom name="task_object_geom_aux2" type="box" pos="0 0 0" size="0.040 0.014 0.018" '
            f'rgba="0.68 0.20 0.74 1" {common}/>',
            f'      <geom name="task_object_geom_aux3" type="box" pos="0 -0.049 0" size="0.040 0.016 0.018" '
            f'rgba="0.68 0.20 0.74 1" {common}/>',
        ]
    elif variant == "peg_round":
        real = [
            f'      <geom name="task_object_geom" type="cylinder" size="0.017 0.055" '
            f'rgba="0.88 0.46 0.08 1" {common}/>'
        ]
    elif variant == "peg_square":
        real = [
            f'      <geom name="task_object_geom" type="box" size="0.016 0.016 0.055" '
            f'rgba="0.20 0.68 0.26 1" {common}/>'
        ]
    else:
        raise ValueError(f"Unknown object variant: {variant}")
    return "\n".join(real + [_dummy_object_geoms(len(real))])


def _goal_body(task_family: str, variant: str) -> str:
    if task_family == "pick_place":
        return """    <body name="basket" pos="0.57 0.16 0.505">
      <geom name="basket_bottom" type="box" pos="0 0 -0.066" size="0.095 0.075 0.008" rgba="0.45 0.28 0.12 1"/>
      <geom type="box" pos="0.095 0 -0.025" size="0.008 0.075 0.043" rgba="0.52 0.33 0.14 1"/>
      <geom type="box" pos="-0.095 0 -0.025" size="0.008 0.075 0.043" rgba="0.52 0.33 0.14 1"/>
      <geom type="box" pos="0 0.075 -0.025" size="0.095 0.008 0.043" rgba="0.52 0.33 0.14 1"/>
      <geom type="box" pos="0 -0.075 -0.025" size="0.095 0.008 0.043" rgba="0.52 0.33 0.14 1"/>
    </body>"""
    if task_family == "push_shape":
        color = "0.68 0.20 0.74 0.28" if variant == "push_b" else "0.12 0.48 0.82 0.28"
        return f"""    <body name="goal_outline" pos="0.57 0.16 0.452">
      <geom name="goal_outline_geom" type="cylinder" size="0.075 0.002" rgba="{color}"
        contype="0" conaffinity="0"/>
    </body>"""
    if task_family == "peg_insertion":
        half_gap = "0.019" if variant == "peg_square" else "0.020"
        return f"""    <body name="insertion_hole" pos="0.53 0.15 0.515">
      <geom name="hole_plate" type="box" pos="0 0 -0.078" size="0.075 0.075 0.006" rgba="0.30 0.32 0.35 0.45" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.050 0 0" size="0.030 0.075 0.040" rgba="0.38 0.40 0.44 1"/>
      <geom type="box" pos="-0.050 0 0" size="0.030 0.075 0.040" rgba="0.38 0.40 0.44 1"/>
      <geom type="box" pos="0 0.050 0" size="{half_gap} 0.030 0.040" rgba="0.38 0.40 0.44 1"/>
      <geom type="box" pos="0 -0.050 0" size="{half_gap} 0.030 0.040" rgba="0.38 0.40 0.44 1"/>
    </body>"""
    raise ValueError(f"Unknown task family: {task_family}")


def _scene_xml(task_family: str, variant: str, scene_id: str) -> str:
    try:
        appearance = SCENE_APPEARANCES[scene_id]
    except KeyError as exc:
        raise ValueError(f"Unknown scene appearance: {scene_id}") from exc
    object_z = "0.485" if task_family == "peg_insertion" else "0.470"
    return f"""<mujoco model="mi_reward_{variant}_{scene_id}">
  <include file="panda_menagerie_adapted.xml"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="{appearance['headlight_diffuse']}" ambient="{appearance['headlight_ambient']}" specular="0.1 0.1 0.1"/>
    <global azimuth="125" elevation="-22"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="{appearance['sky_rgb1']}" rgb2="{appearance['sky_rgb2']}" width="256" height="1024"/>
    <texture name="table_tex" type="2d" builtin="checker" rgb1="{appearance['table_rgb1']}" rgb2="{appearance['table_rgb2']}" width="256" height="256"/>
    <material name="table_mat" texture="table_tex" texrepeat="4 4" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light name="key_light" pos="{appearance['light_pos']}" dir="0.2 0.2 -1" directional="true"/>
    <!-- Fixed third-person table camera. The complete 0.8m work surface is
         visible while the Panda base remains outside the primary framing. -->
    <camera name="global_cam" pos="1.05 -0.90 1.45" xyaxes="0.866 0.500 0 -0.351 0.608 0.712" fovy="42"/>
    <geom name="floor_geom" type="plane" size="2 2 0.05" rgba="{appearance['floor_rgba']}"/>
    <body name="table" pos="0.53 0 0.38">
      <geom name="table_geom" type="box" size="0.40 0.40 0.05" material="table_mat"/>
    </body>
    <body name="wall" pos="0.46 0.42 0.67">
      <geom name="wall_geom" type="box" size="0.46 0.02 0.24" rgba="{appearance['wall_rgba']}"/>
    </body>
    <body name="ee_target" mocap="true" pos="0.5545 0 0.6245" quat="0 0.70714 0.70707 0">
      <body name="panda_hand">
        <geom name="proxy_palm_geom" type="box" pos="0 0 -0.060" size="0.036 0.030 0.014"
          rgba="0.16 0.18 0.20 1" contype="0" conaffinity="0"/>
        <body name="proxy_left_finger">
          <joint name="proxy_left_finger_joint" type="slide" axis="0 1 0" range="0 0.05" damping="4"/>
          <geom name="left_finger_geom" type="box" pos="0 0 -0.018" size="0.009 0.008 0.042"
            rgba="0.12 0.14 0.16 1" friction="1.5 0.01 0.001" mass="0.025"/>
        </body>
        <body name="proxy_right_finger">
          <joint name="proxy_right_finger_joint" type="slide" axis="0 -1 0" range="0 0.05" damping="4"/>
          <geom name="right_finger_geom" type="box" pos="0 0 -0.018" size="0.009 0.008 0.042"
            rgba="0.12 0.14 0.16 1" friction="1.5 0.01 0.001" mass="0.025"/>
        </body>
      </body>
    </body>
    <body name="task_object" pos="0.48 -0.16 {object_z}">
      <freejoint name="task_object_freejoint"/>
{_object_geoms(variant)}
    </body>
{_goal_body(task_family, variant)}
  </worldbody>
  <tendon>
    <fixed name="proxy_gripper_split">
      <joint joint="proxy_left_finger_joint" coef="0.5"/>
      <joint joint="proxy_right_finger_joint" coef="0.5"/>
    </fixed>
  </tendon>
  <equality>
    <joint joint1="proxy_left_finger_joint" joint2="proxy_right_finger_joint"/>
  </equality>
  <actuator>
    <position name="gripper" tendon="proxy_gripper_split" kp="1000" kv="40"
      ctrlrange="0 0.05" forcerange="-200 200"/>
  </actuator>
  <keyframe>
    <key name="home"
      qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0.05 0.05 0.48 -0.16 {object_z} 1 0 0 0"
      ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 1"/>
  </keyframe>
</mujoco>
"""


def _revision(menagerie_root: Path) -> str | None:
    repo = menagerie_root.parent
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate(scene_paths: Iterable[Path]) -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo validation requested, but the mujoco package is not installed") from exc

    required = {
        "body": ["panda", "panda_hand", "ee_target", "task_object", "table"],
        "geom": [
            "table_geom",
            "wall_geom",
            "task_object_geom",
            "left_finger_geom",
            "right_finger_geom",
        ],
        "actuator": ["gripper"],
        "camera": ["global_cam"],
        "key": ["home"],
    }
    object_types = {
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        "camera": mujoco.mjtObj.mjOBJ_CAMERA,
        "key": mujoco.mjtObj.mjOBJ_KEY,
    }
    for path in scene_paths:
        model = mujoco.MjModel.from_xml_path(str(path))
        for kind, names in required.items():
            for name in names:
                if mujoco.mj_name2id(model, object_types[kind], name) < 0:
                    raise RuntimeError(f"Generated scene {path} is missing {kind} {name!r}")


def prepare_assets(menagerie_root: Path, output_root: Path, *, validate: bool) -> list[Path]:
    source = menagerie_root / "panda.xml"
    mesh_dir = menagerie_root / "assets"
    license_path = menagerie_root / "LICENSE"
    for required in (source, mesh_dir, license_path):
        if not required.exists():
            raise FileNotFoundError(f"Required MuJoCo Menagerie asset is missing: {required}")

    output_root.mkdir(parents=True, exist_ok=True)
    adapted = output_root / "panda_menagerie_adapted.xml"
    _adapt_panda(source, adapted, mesh_dir)

    scene_paths = []
    for filename, (task_family, variant, scene_id) in SCENE_VARIANTS.items():
        path = output_root / filename
        path.write_text(_scene_xml(task_family, variant, scene_id), encoding="utf-8")
        scene_paths.append(path)

    manifest = {
        "robot_source": MENAGERIE_URL,
        "robot_revision": _revision(menagerie_root),
        "robot_license": str(license_path.resolve()),
        "robot_model": str(source.resolve()),
        "generated_scenes": [str(path.resolve()) for path in scene_paths],
        "scene_appearances": list(SCENE_APPEARANCES),
        "task_geometry": "Generated from native MuJoCo primitives by prepare_mujoco_assets.py",
    }
    (output_root / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if validate:
        _validate(scene_paths)
    return scene_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--menagerie-root",
        default="assets/vendor/mujoco_menagerie/franka_emika_panda",
        type=Path,
    )
    parser.add_argument("--output-root", default="assets/custom_task/scene", type=Path)
    parser.add_argument("--no-validate", action="store_true", help="Skip loading generated scenes with MuJoCo.")
    args = parser.parse_args()
    paths = prepare_assets(args.menagerie_root.resolve(), args.output_root.resolve(), validate=not args.no_validate)
    status = "generated" if args.no_validate else "generated and validated"
    print(f"[assets] {len(paths)} MuJoCo scenes {status} in {args.output_root}")


if __name__ == "__main__":
    main()
