"""Tests for Claude judge."""

import pytest
from src.judge.prompts import JudgePrompts


def test_create_judgment_prompt():
    """Test prompt creation."""
    dimensions = [
        {"name": "accuracy", "description": "Is it correct?", "weight": 1.0}
    ]

    prompt = JudgePrompts.create_judgment_prompt(
        context="Test context",
        student_message="Test message",
        tutor_response="Test response",
        dimensions=dimensions
    )

    assert "Test context" in prompt
    assert "Test message" in prompt
    assert "Test response" in prompt
    assert "accuracy" in prompt
