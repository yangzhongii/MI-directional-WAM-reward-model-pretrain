"""CPU-only checks for the official Robometer processed-cache bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mi_reward.data.robometer_ingest import _materialize_frames


def test_processed_npz_frames_are_materialized_as_pngs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    frames = np.zeros((3, 6, 8, 3), dtype=np.uint8)
    frames[1, ..., 1] = 255
    np.savez_compressed(dataset_root / "frames.npz", frames=frames)

    paths = _materialize_frames("frames.npz", tmp_path / "materialized", dataset_root)

    assert len(paths) == 3
    assert all(Path(path).is_file() for path in paths)
    assert np.asarray(Image.open(paths[1]))[0, 0, 1] == 255
