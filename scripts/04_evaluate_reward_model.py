#!/usr/bin/env python3
"""Evaluate trained reward model."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.reward_model.inference import RewardModelScorer
from src.reward_model.evaluator import RewardModelEvaluator
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__, "outputs/logs/evaluation.log")


def main(args):
    """Evaluate reward model."""
    print("=" * 60)
    print("Reward Model Evaluation")
    print("=" * 60)
    print()

    # Load model
    print(f"Loading model from {args.model_path}...")
    scorer = RewardModelScorer(model_path=args.model_path)
    print("✓ Model loaded")
    print()

    # Evaluate
    print(f"Evaluating on {args.judgments_file}...")
    evaluator = RewardModelEvaluator(
        scorer=scorer,
        judgments_path=args.judgments_file
    )

    metrics = evaluator.evaluate(output_dir=args.output_dir)

    # Summary
    print()
    print("=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"  Pearson correlation: {metrics['pearson']:.4f}")
    print(f"  Spearman correlation: {metrics['spearman']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print()
    print(f"  Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate reward model performance"
    )
    parser.add_argument(
        "--model-path",
        default="models/reward_model_final",
        help="Path to trained model"
    )
    parser.add_argument(
        "--judgments-file",
        default="data/judgments/claude_judgments.json",
        help="Path to Claude judgments file"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/metrics",
        help="Directory to save evaluation results"
    )

    args = parser.parse_args()
    main(args)
