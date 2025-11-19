"""Inference utilities for reward model."""

import torch
from pathlib import Path
from typing import List, Union
from transformers import AutoTokenizer
import logging

from src.reward_model.model import RewardModel

logger = logging.getLogger(__name__)


class RewardModelScorer:
    """Score tutor responses using trained reward model."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto"
    ):
        """Initialize scorer.

        Args:
            model_path: Path to saved model directory
            device: Device to use ('auto', 'cuda', or 'cpu')
        """
        self.model_path = Path(model_path)

        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Loading reward model from {model_path}")
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

        self.model.eval()
        self.model.to(self.device)

        logger.info(f"Reward model loaded on {self.device}")

    def _load_model(self) -> RewardModel:
        """Load trained model."""
        import json

        # Load config
        with open(self.model_path / 'config.json', 'r') as f:
            config = json.load(f)

        # Initialize model
        model = RewardModel(
            base_model_name=config['base_model'],
            hidden_size=config.get('hidden_size'),
            max_length=config.get('max_length', 512)
        )

        # Load weights
        state_dict = torch.load(
            self.model_path / 'model.pt',
            map_location=self.device
        )
        model.load_state_dict(state_dict)

        return model

    def _load_tokenizer(self):
        """Load tokenizer."""
        return AutoTokenizer.from_pretrained(str(self.model_path))

    def score(
        self,
        context: str,
        response: str
    ) -> float:
        """Score a single response.

        Args:
            context: Conversation context
            response: Tutor response to score

        Returns:
            Quality score (1-10 scale)
        """
        # Create input text
        input_text = f"Context: {context}\nTutor: {response}"

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Inference
        with torch.no_grad():
            score = self.model(input_ids, attention_mask)

        return float(score.item())

    def score_batch(
        self,
        contexts: List[str],
        responses: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """Score multiple responses in batches.

        Args:
            contexts: List of conversation contexts
            responses: List of tutor responses
            batch_size: Batch size for inference

        Returns:
            List of quality scores
        """
        if len(contexts) != len(responses):
            raise ValueError("contexts and responses must have same length")

        all_scores = []

        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]

            # Create input texts
            input_texts = [
                f"Context: {ctx}\nTutor: {resp}"
                for ctx, resp in zip(batch_contexts, batch_responses)
            ]

            # Tokenize
            encoding = self.tokenizer(
                input_texts,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # Inference
            with torch.no_grad():
                scores = self.model(input_ids, attention_mask)

            all_scores.extend(scores.squeeze(-1).cpu().numpy().tolist())

        return all_scores
