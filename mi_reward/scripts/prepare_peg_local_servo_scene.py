"""Derive the isolated P1 local-servo scene without touching legacy assets."""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def derive(source: Path, target: Path, clearance_mm: float) -> None:
    root = ET.parse(source).getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError("scene has no worldbody")
    if world.find("camera[@name='wrist_cam']") is None:
        # The proxy hand is at the origin of panda_hand.  This camera is fixed
        # relative to that body and therefore moves with the grasped peg.
        hand = world.find("body[@name='ee_target']/body[@name='panda_hand']")
        if hand is None:
            raise ValueError("scene has no panda_hand proxy body")
        hand.append(ET.fromstring(
            '<camera name="wrist_cam" pos="0 -0.085 0.045" '
            'quat="0.313903 0.686141 0.463055 0.465032" fovy="48"/>'
        ))
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    if equality.find("weld[@name='peg_grasp_weld']") is None:
        ET.SubElement(equality, "weld", {
            "name": "peg_grasp_weld",
            "body1": "panda_hand",
            "body2": "task_object",
            # Preserve the source scene's initial grasp offset instead of
            # collapsing both body origins onto one another.
            # Relative pose is expressed in the rotated panda_hand frame.
            "relpose": "-0.1600 -0.0745 0.1395 0 -0.7071 -0.7071 0",
            "solref": "0.01 1",
            "solimp": "0.95 0.99 0.01",
        })
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    if contact.find("exclude[@name='grasp_internal_collision']") is None:
        ET.SubElement(contact, "exclude", {
            "name": "grasp_internal_collision",
            "body1": "panda_hand",
            "body2": "task_object",
        })
    if contact.find("exclude[@name='finger_internal_collision']") is None:
        ET.SubElement(contact, "exclude", {
            "name": "finger_internal_collision",
            "body1": "proxy_left_finger",
            "body2": "proxy_right_finger",
        })
    if contact.find("exclude[@name='robot_proxy_collision']") is None:
        ET.SubElement(contact, "exclude", {
            "name": "robot_proxy_collision",
            "body1": "panda",
            "body2": "ee_target",
        })
    robot_bodies = ["panda", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"]
    for first in robot_bodies:
        for second in robot_bodies:
            if first >= second:
                continue
            name = f"robot_self_{first}_{second}"
            if contact.find(f"exclude[@name='{name}']") is None:
                ET.SubElement(contact, "exclude", {"name": name, "body1": first, "body2": second})
    # The included Panda visual/collision mesh is only a static reference in
    # this proxy-hand pilot. Disable its self-collision contacts so they cannot
    # masquerade as peg/socket forces; proxy fingers and task geoms remain
    # physical.
    for body in root.findall(".//body"):
        if body.get("name") in {"panda", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"}:
            for geom in body.findall(".//geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        if body.get("name") in {"panda_hand", "proxy_left_finger", "proxy_right_finger"}:
            for geom in body.findall(".//geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
    root.set("model", root.get("model", "peg") + "_local_servo")
    comment = ET.Comment(f"P1 metadata: clearance_mm={clearance_mm:.3f}; physical grasp weld")
    root.insert(0, comment)
    ET.indent(root, space="  ")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ET.tostring(root, encoding="unicode") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--clearance-mm", type=float, default=1.0)
    args = parser.parse_args()
    derive(args.source, args.target, args.clearance_mm)
    print(f"wrote {args.target} (clearance_mm={args.clearance_mm})")


if __name__ == "__main__":
    main()
