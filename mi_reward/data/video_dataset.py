from __future__ import annotations

from pathlib import Path
from typing import Iterator

from torch.utils.data import Dataset

from mi_reward.data.schema import TrajectoryExample, read_jsonl


class TrajectoryManifestDataset(Dataset[TrajectoryExample]):
    def __init__(self, manifest_path: str | Path):
        self.examples = read_jsonl(manifest_path, TrajectoryExample)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrajectoryExample:
        return self.examples[index]

    def by_task(self) -> dict[str, list[TrajectoryExample]]:
        grouped: dict[str, list[TrajectoryExample]] = {}
        for example in self.examples:
            grouped.setdefault(example.task, []).append(example)
        return grouped

    def __iter__(self) -> Iterator[TrajectoryExample]:
        return iter(self.examples)
