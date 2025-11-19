"""Tests for reward model."""

import pytest
import torch
from src.reward_model.model import RewardModel


def test_reward_model_forward():
    """Test model forward pass."""
    model = RewardModel()

    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    scores = model(input_ids, attention_mask)

    assert scores.shape == (batch_size, 1)
