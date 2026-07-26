from __future__ import annotations

from config import TrainingConfig


def main() -> None:
    config = TrainingConfig.cfg.load_json("train.json")

    print(f"optimizer.name: {config.optimizer.name}")
    print(f"optimizer.lr: {config.optimizer.lr}")
    print(f"seed: {config.seed}")

    # OptimizerConfig is a ConfigNode too, so it fingerprints its own section.
    print(f"optimizer id: {config.optimizer.cfg.hash()}")

    run_id = config.cfg.hash()
    print(f"run id: {run_id}")

    resolved = config.cfg.save_json(config.output_dir / run_id / "resolved.json")
    print(f"saved: {resolved}")


if __name__ == "__main__":
    main()
