"""Register the project LIBERO experiment, then run NVIDIA's official trainer."""

from world_model.training.cosmos_action_lora_official import register_experiment


register_experiment()

from cosmos_oss.scripts.train import main  # noqa: E402


if __name__ == "__main__":
    main()

