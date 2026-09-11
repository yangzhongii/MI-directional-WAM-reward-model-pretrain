"""Small two-node planner + MuJoCo smoke test for the rsync transport."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

from mi_reward.data.instance_orchestrator import reports_as_dict, run_configured_stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default="zhonghaoyang@172.16.0.3")
    parser.add_argument(
        "--remote-root",
        default="/mnt/public/zhonghaoyang/MI-directional-WAM-reward-model-pretrain",
    )
    parser.add_argument(
        "--remote-python",
        default="/mnt/public/zhonghaoyang/MI-directional-WAM-reward-model-pretrain/.venv/bin/python",
    )
    parser.add_argument("--output-root", default="logs/mi_reward/distributed_smoke")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    output_root = (root / args.output_root).resolve()
    source = output_root / "base_records.jsonl"
    task_config = root / "mi_reward/configs/tasks/pick_place_apple_banana.yaml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "".join(
            json.dumps(
                {
                    "base_id": f"distributed_smoke/pick_place/seed-{index:04d}",
                    "task": "pick up the task object and place it in the basket",
                    "task_family": "pick_place",
                    "physical_task_config": str(task_config),
                    "metadata": {"simulator_reference_seed": False},
                }
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )

    local_python = root / ".venv/bin/python"
    config = {
        "generation": {
            "planner_candidates": 1,
            "planner_seed": 9100,
            "render_width": 160,
            "render_height": 128,
        },
        "execution": {
            "distributed": {
                "enabled": True,
                "transport": "rsync",
                "sync_project": True,
                "stages": ["planner", "simulator"],
                "nodes": [
                    {
                        "host": "local",
                        "rank": 0,
                        "repo_root": str(root),
                        "python": str(local_python),
                        "env": {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
                    },
                    {
                        "host": args.remote_host,
                        "rank": 1,
                        "repo_root": args.remote_root,
                        "python": args.remote_python,
                        "env": {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
                    },
                ],
            },
            "stages": [
                {
                    "name": "planner",
                    "input_records": str(source),
                    "output_records": str(output_root / "planner_records.jsonl"),
                    "command": [
                        "{python}", "-m", "mi_reward.planning.rigid_task_planner",
                        "--request", "{request}", "--result", "{result}",
                        "--num-candidates", "{planner_candidates}", "--seed", "{planner_seed}",
                    ],
                },
                {
                    "name": "simulator",
                    "output_records": str(output_root / "physical_records.jsonl"),
                    "command": [
                        "{python}", "-m", "mi_reward.sim.mujoco_generalization_worker",
                        "--request", "{request}", "--result", "{result}",
                        "--width", "{render_width}", "--height", "{render_height}",
                    ],
                },
            ],
        },
    }

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    reports = run_configured_stages(config, root=root)
    final_rows = [
        json.loads(line)
        for line in (output_root / "physical_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rank_counts = {}
    for rank in range(2):
        shard = output_root / ".distributed/simulator" / f"output.rank{rank}.jsonl"
        rank_counts[str(rank)] = sum(1 for line in shard.read_text(encoding="utf-8").splitlines() if line.strip())
    print(
        json.dumps(
            {
                "status": "passed",
                "controller": socket.gethostname(),
                "reports": reports_as_dict(reports),
                "records_per_rank": rank_counts,
                "final_records": len(final_rows),
                "output": str(output_root / "physical_records.jsonl"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
