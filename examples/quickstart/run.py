from __future__ import annotations

from config import TrainingConfig


def main() -> None:
    config = TrainingConfig.load_json("train.json")

    print(f"optimizer.name: {config.optimizer.name}")
    print(f"optimizer.lr: {config.optimizer.lr}")
    print(f"seed: {config.seed}")

    run_id = config.config_hash()
    print(f"run id: {run_id}")

    resolved = config.save_json(config.output_dir / run_id / "resolved.json")
    print(f"saved: {resolved}")


if __name__ == "__main__":
    main()
