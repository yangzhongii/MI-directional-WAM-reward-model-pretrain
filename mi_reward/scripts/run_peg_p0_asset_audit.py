"""Audit the existing peg scene against the frozen P1 physical prerequisites.

This deliberately reports missing prerequisites instead of silently treating the
legacy kinematic rollout as a physical peg-insertion benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mi_reward/configs/tasks/peg_insertion_variants.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    cfg_path = root / args.config
    cfg = yaml.safe_load(cfg_path.read_text())
    sim = cfg["simulation"]
    model_path = root / sim["model_path"]
    xml = ET.parse(model_path).getroot()
    cameras = {c.get("name") for c in xml.findall(".//camera")}
    checks = {
        "wrist_camera_present": "wrist_cam" in cameras,
        "eye_in_hand_camera_configured": sim.get("camera_name") == "wrist_cam",
        "legacy_global_camera_only": sim.get("camera_name") == "global_cam",
        "direct_qpos_attachment_disabled": not bool(sim.get("kinematic_task_proxy", False)),
        "contact_force_logging_configured": bool(sim.get("contact_force_logging", False)),
        "clearance_mm_configured": "clearance_mm" in sim,
    }
    report = {
        "benchmark": "peg_insertion_local_servo",
        "stage": "P0_asset_physics_audit",
        "config": str(cfg_path),
        "model": str(model_path),
        "cameras": sorted(cameras),
        "checks": checks,
        "p0_pass": all(
            checks[k]
            for k in (
                "wrist_camera_present",
                "eye_in_hand_camera_configured",
                "direct_qpos_attachment_disabled",
                "contact_force_logging_configured",
                "clearance_mm_configured",
            )
        ),
        "interpretation": (
            "Existing legacy scene is not yet a valid P0 physical-contact benchmark."
            if not all(checks[k] for k in (
                "wrist_camera_present", "eye_in_hand_camera_configured",
                "direct_qpos_attachment_disabled", "contact_force_logging_configured",
                "clearance_mm_configured"))
            else "P0 structural prerequisites are present."
        ),
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
