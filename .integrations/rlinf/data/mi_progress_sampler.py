"""MI-progress-guided stratified replay sampling.

DROP-IN LOCATION: rlinf/data/mi_progress_sampler.py

Provides MIProgressStratifiedSampler for Phase B: stratifies online
transitions by MI potential delta into positive/neutral/negative classes.

Preserves RLPD's 50/50 demo-online mixing.
Only stratifies the online half.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch

logger = logging.getLogger(__name__)


class MIProgressStratifiedSampler:
    """Stratified sampler for MI-progress-guided replay.

    Classifies online transitions by mi_delta and samples configured
    fractions from each class. Falls back to uniform when a class
    has insufficient transitions.

    Does NOT modify SAC/RLPD loss, critic target, or actor objective.
    """

    def __init__(
        self,
        positive_threshold: float = 0.02,
        negative_threshold: float = -0.02,
        min_confidence: float = 0.2,
        positive_fraction: float = 0.4,
        neutral_fraction: float = 0.2,
        negative_fraction: float = 0.4,
        fallback_to_uniform: bool = True,
        max_resample_attempts: int = 5,
    ):
        """
        Args:
            positive_threshold: mi_delta > this → positive progress
            negative_threshold: mi_delta < this → regression
            min_confidence: minimum confidence for stratified sampling
            positive_fraction: target fraction of positive transitions
            neutral_fraction: target fraction of neutral transitions
            negative_fraction: target fraction of negative transitions
            fallback_to_uniform: fall back to uniform if class is empty
            max_resample_attempts: max attempts before fallback
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.min_confidence = min_confidence
        self.positive_fraction = positive_fraction
        self.neutral_fraction = neutral_fraction
        self.negative_fraction = negative_fraction
        self.fallback_to_uniform = fallback_to_uniform
        self.max_resample_attempts = max_resample_attempts

        # Validate fractions
        total = positive_fraction + neutral_fraction + negative_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Stratified fractions must sum to 1.0, got {total}")

    def classify(self, mi_delta: float, mi_confidence: float) -> str:
        """Classify a single transition by MI progress.

        Returns:
            "positive", "neutral", "negative", or "low_confidence"
        """
        if mi_confidence < self.min_confidence:
            return "low_confidence"
        if mi_delta > self.positive_threshold:
            return "positive"
        elif mi_delta < self.negative_threshold:
            return "negative"
        else:
            return "neutral"

    def sample_indices(
        self,
        online_buffer_size: int,
        batch_size: int,
        mi_deltas: torch.Tensor,
        mi_confidences: torch.Tensor,
        indices: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample indices stratified by MI progress class.

        Args:
            online_buffer_size: total size of online buffer
            batch_size: number of online transitions to sample
            mi_deltas: [N] stored mi_delta values
            mi_confidences: [N] stored confidence values
            indices: optional global indices for the online buffer
            seed: optional random seed

        Returns:
            sampled_indices: [batch_size] local indices into online buffer
            stats: dict with actual sampled fractions
        """
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        N = min(online_buffer_size, len(mi_deltas))

        # Classify all online transitions
        classes: dict[str, list[int]] = defaultdict(list)
        for i in range(N):
            cls = self.classify(
                float(mi_deltas[i].item()),
                float(mi_confidences[i].item()),
            )
            classes[cls].append(i)

        # Target counts per class
        target_positive = int(batch_size * self.positive_fraction)
        target_neutral = int(batch_size * self.neutral_fraction)
        target_negative = batch_size - target_positive - target_neutral

        sampled = []
        class_counts = {}

        # Sample from each class
        for cls_name, target_count in [
            ("positive", target_positive),
            ("neutral", target_neutral),
            ("negative", target_negative),
        ]:
            available = classes.get(cls_name, [])
            if len(available) >= target_count:
                perm = torch.randperm(len(available), generator=rng)[:target_count]
                sampled.extend([available[p.item()] for p in perm])
                class_counts[cls_name] = target_count
            elif len(available) > 0 and self.fallback_to_uniform:
                # Take all available
                sampled.extend(available)
                class_counts[cls_name] = len(available)
                # Fill remainder from low-confidence or other classes
                shortage = target_count - len(available)
                fill_pool = []
                for other_cls in ["low_confidence", "positive", "neutral", "negative"]:
                    if other_cls != cls_name and other_cls in classes:
                        fill_pool.extend(classes[other_cls])
                if fill_pool:
                    fill_count = min(shortage, len(fill_pool))
                    perm = torch.randperm(len(fill_pool), generator=rng)[:fill_count]
                    sampled.extend([fill_pool[p.item()] for p in perm])
            else:
                class_counts[cls_name] = 0

        # If total is short, fill with uniform random
        if len(sampled) < batch_size and self.fallback_to_uniform:
            all_available = list(range(N))
            remaining = batch_size - len(sampled)
            # Avoid duplicates
            already = set(sampled)
            pool = [i for i in all_available if i not in already]
            if pool:
                perm = torch.randperm(len(pool), generator=rng)[:remaining]
                sampled.extend([pool[p.item()] for p in perm])

        # Truncate to batch_size
        sampled = sampled[:batch_size]
        sampled_tensor = torch.tensor(sampled, dtype=torch.long)

        # Stats
        stats = {
            "mi_positive_fraction": class_counts.get("positive", 0) / max(len(sampled), 1),
            "mi_neutral_fraction": class_counts.get("neutral", 0) / max(len(sampled), 1),
            "mi_negative_fraction": class_counts.get("negative", 0) / max(len(sampled), 1),
            "mi_low_confidence_fraction": class_counts.get("low_confidence", 0) / max(len(sampled), 1),
            "total_sampled": len(sampled),
            "target_positive": target_positive,
            "target_neutral": target_neutral,
            "target_negative": target_negative,
        }

        return sampled_tensor, stats


# ---------------------------------------------------------------------------
# Integration into RLPD training loop
# ---------------------------------------------------------------------------
#
# In the RLPD trainer's sample step:
#
#   # 50% demo, 50% online (unchanged)
#   demo_batch = demo_buffer.sample(batch_size // 2)
#
#   if mi_guided_replay.enabled:
#       online_indices, mi_stats = mi_sampler.sample_indices(
#           online_buffer_size=len(online_buffer),
#           batch_size=batch_size // 2,
#           mi_deltas=online_buffer.get_field("mi_delta"),
#           mi_confidences=online_buffer.get_field("mi_confidence"),
#       )
#       online_batch = online_buffer.sample_indices(online_indices)
#       log_mi_stats(mi_stats)
#   else:
#       online_batch = online_buffer.sample(batch_size // 2)
#
#   batch = merge(demo_batch, online_batch)
# ---------------------------------------------------------------------------
