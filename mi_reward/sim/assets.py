"""Versioned rigid-task asset catalog used by simulator adapters.

The catalog intentionally stores URIs rather than importing a particular
robotics asset package.  A deployment can map ``builtin://`` entries to its
MuJoCo XML/mesh tree while keeping the reward manifest portable.
"""

from __future__ import annotations

from dataclasses import dataclass

from mi_reward.sim.base import InstanceAsset


CATALOG_VERSION = "rigid_v1"


@dataclass(frozen=True)
class CatalogAsset:
    task_family: str
    category: str
    task_role: str
    asset: InstanceAsset


_ASSETS = (
    CatalogAsset("pick_place", "apple", "source_object", InstanceAsset("apple", "builtin://pick_place/apple")),
    CatalogAsset("pick_place", "banana", "target_object", InstanceAsset("banana", "builtin://pick_place/banana")),
    CatalogAsset("pick_place", "basket", "goal_container", InstanceAsset("basket", "builtin://pick_place/basket")),
    CatalogAsset("push_shape", "push_t", "source_object", InstanceAsset("push_t", "builtin://push_shape/push_t")),
    CatalogAsset("push_shape", "push_b", "target_object", InstanceAsset("push_b", "builtin://push_shape/push_b")),
    CatalogAsset(
        "peg_insertion", "peg_round", "source_object", InstanceAsset("peg_round", "builtin://peg_insertion/peg_round")
    ),
    CatalogAsset(
        "peg_insertion", "peg_square", "target_object", InstanceAsset("peg_square", "builtin://peg_insertion/peg_square")
    ),
    CatalogAsset(
        "peg_insertion", "hole_round", "goal", InstanceAsset("hole_round", "builtin://peg_insertion/hole_round")
    ),
    CatalogAsset(
        "peg_insertion", "hole_square", "goal", InstanceAsset("hole_square", "builtin://peg_insertion/hole_square")
    ),
)


def list_assets(task_family: str | None = None) -> tuple[CatalogAsset, ...]:
    """Return the immutable catalog entries, optionally filtered by task."""

    if task_family is None:
        return _ASSETS
    return tuple(item for item in _ASSETS if item.task_family == task_family)


def describe_asset(task_family: str, category: str) -> CatalogAsset:
    """Resolve a catalog entry, including its task role and source URI."""

    for item in _ASSETS:
        if item.task_family == task_family and item.category == category:
            return item
    known = ", ".join(item.category for item in list_assets(task_family)) or "<none>"
    raise KeyError(f"Unknown asset {category!r} for task_family={task_family!r}; known: {known}")


def load_asset(task_family: str, category: str) -> InstanceAsset:
    """Return the simulator-ready asset for ``SimulatorBackend.apply_instance_variant``."""

    return describe_asset(task_family, category).asset
