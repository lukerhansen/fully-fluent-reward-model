"""Evaluate reward model performance."""

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from src.utils.metrics import compute_regression_metrics

logger = logging.getLogger(__name__)


class RewardModelEvaluator:
    """Evaluate reward model against Claude judgments."""

    def __init__(self, scorer, judgments_path: str):
        """Initialize evaluator.

        Args:
            scorer: RewardModelScorer instance
            judgments_path: Path to Claude judgments
        """
        self.scorer = scorer

        with open(judgments_path, 'r') as f:
            self.judgments = json.load(f)

    def evaluate(self, output_dir: str = "outputs/metrics") -> Dict:
        """Evaluate model performance.

        Args:
            output_dir: Directory to save evaluation results

        Returns:
            Evaluation metrics dictionary
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Evaluating on {len(self.judgments)} judgments...")

        # Get predictions and targets
        contexts = []
        responses = []
        claude_scores = []

        for item in self.judgments:
            contexts.append(item['context'])
            responses.append(item['tutor_response'])
            claude_scores.append(item['judgment']['overall']['score'])

        # Batch score
        model_scores = self.scorer.score_batch(contexts, responses)

        # Compute metrics
        metrics = compute_regression_metrics(
            np.array(claude_scores),
            np.array(model_scores)
        )

        logger.info(f"Evaluation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        # Save metrics
        with open(Path(output_dir) / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create visualizations
        self._create_visualizations(
            claude_scores, model_scores, output_dir
        )

        # Check minimum correlation
        if metrics['pearson'] < 0.7:
            logger.warning(
                f"⚠️  Pearson correlation ({metrics['pearson']:.3f}) is below "
                f"recommended threshold of 0.7. Consider:"
                f"\n  - Collecting more training data"
                f"\n  - Adjusting model architecture"
                f"\n  - Tuning hyperparameters"
            )
        else:
            logger.info(
                f"✓ Pearson correlation ({metrics['pearson']:.3f}) meets "
                f"quality threshold"
            )

        return metrics

    def _create_visualizations(
        self,
        claude_scores: List[float],
        model_scores: List[float],
        output_dir: str
    ) -> None:
        """Create evaluation visualizations."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter plot
        axes[0].scatter(claude_scores, model_scores, alpha=0.6)
        axes[0].plot([1, 10], [1, 10], 'r--', label='Perfect alignment')
        axes[0].set_xlabel('Claude Scores')
        axes[0].set_ylabel('Model Scores')
        axes[0].set_title('Model vs Claude Scores')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Distribution comparison
        axes[1].hist(claude_scores, bins=20, alpha=0.5, label='Claude', density=True)
        axes[1].hist(model_scores, bins=20, alpha=0.5, label='Model', density=True)
        axes[1].set_xlabel('Score')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Score Distributions')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'evaluation_plots.png', dpi=150)
        plt.close()

        logger.info(f"Saved visualizations to {output_dir}/evaluation_plots.png")
