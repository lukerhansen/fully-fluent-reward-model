"""Fully Fluent Reward Model.

Train reward models from Claude judgments for language tutoring evaluation.
"""

__version__ = "0.1.0"

from src.data.loader import ConversationLoader
from src.judge.claude_judge import ClaudeJudge
from src.reward_model.inference import RewardModelScorer
from src.reward_model.trainer import RewardModelTrainer

__all__ = [
    "ConversationLoader",
    "ClaudeJudge",
    "RewardModelScorer",
    "RewardModelTrainer",
]
