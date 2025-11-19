#!/usr/bin/env python3
"""Train reward model from Claude judgments."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.reward_model.trainer import RewardModelTrainer
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__, "outputs/logs/training.log")


def main(args):
    """Train reward model."""
    print("=" * 60)
    print("Reward Model Training")
    print("=" * 60)
    print()

    # Initialize trainer
    print("Initializing trainer...")
    trainer = RewardModelTrainer(config_path=args.config)
    print("✓ Trainer initialized")
    print()

    # Prepare data
    print(f"Preparing data from {args.judgments_file}...")
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(
        judgments_path=args.judgments_file
    )
    print(f"✓ Data prepared:")
    print(f"    Train: {len(train_dataset)} samples")
    print(f"    Val: {len(val_dataset)} samples")
    print(f"    Test: {len(test_dataset)} samples")
    print()

    # Train
    print("Training model...")
    print("This may take 10-30 minutes depending on data size...")
    print()

    metrics = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_path=args.output_path
    )

    # Summary
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Best validation loss: {metrics['best_val_loss']:.4f}")
    print(f"  Final epoch: {metrics['final_epoch']}")
    print(f"  Model saved to: {args.output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train reward model from judgments"
    )
    parser.add_argument(
        "--judgments-file",
        default="data/judgments/claude_judgments.json",
        help="Path to Claude judgments file"
    )
    parser.add_argument(
        "--output-path",
        default="models/reward_model_final",
        help="Path to save trained model"
    )
    parser.add_argument(
        "--config",
        default="config/reward_model_config.yaml",
        help="Training configuration file"
    )

    args = parser.parse_args()
    main(args)
