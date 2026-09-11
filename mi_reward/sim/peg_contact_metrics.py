"""MuJoCo contact force summaries for the P1 peg local-servo pilot."""
from __future__ import annotations

from typing import Any

import numpy as np


def contact_force_summary(mujoco: Any, model: Any, data: Any) -> dict[str, float | int]:
    """Return normal/tangential contact force totals for the current frame."""
    normal = 0.0
    tangential = 0.0
    total = 0.0
    count = int(data.ncon)
    wrench = np.zeros(6, dtype=np.float64)
    for index in range(count):
        mujoco.mj_contactForce(model, data, index, wrench)
        normal += abs(float(wrench[0]))
        tangential += float(np.linalg.norm(wrench[1:3]))
        total += float(np.linalg.norm(wrench[:3]))
    return {
        "contact_count": count,
        "normal_force_n": normal,
        "tangential_force_n": tangential,
        "resultant_force_n": total,
    }
