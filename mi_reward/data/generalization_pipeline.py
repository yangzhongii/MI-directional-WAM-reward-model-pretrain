"""Public CLI for scene/instance generalization data generation and ingestion."""

from mi_reward.data.instance_pipeline import (
    main,
    validate_config,
    validate_training_manifest,
)

__all__ = ["main", "validate_config", "validate_training_manifest"]


if __name__ == "__main__":
    main()
