"""Reward model architecture."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional


class RewardModel(nn.Module):
    """Transformer-based reward model for scoring tutor responses."""

    def __init__(
        self,
        base_model_name: str = "distilbert-base-uncased",
        hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        max_length: int = 512
    ):
        """Initialize reward model.

        Args:
            base_model_name: HuggingFace model name
            hidden_size: Hidden layer size (uses model default if None)
            dropout: Dropout probability
            max_length: Maximum sequence length
        """
        super().__init__()

        self.config = AutoConfig.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)

        hidden_size = hidden_size or self.config.hidden_size
        self.max_length = max_length

        # Regression head for scoring
        self.score_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Single score output
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Scores [batch_size, 1]
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Compute score
        score = self.score_head(cls_output)  # [batch_size, 1]

        return score

    def get_config(self) -> Dict:
        """Get model configuration.

        Returns:
            Configuration dictionary
        """
        return {
            'base_model': self.config.name_or_path,
            'hidden_size': self.config.hidden_size,
            'max_length': self.max_length,
            'num_parameters': sum(p.numel() for p in self.parameters()),
        }
