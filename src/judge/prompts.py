"""Prompt templates for Claude judge."""

from typing import Dict, List


class JudgePrompts:
    """Prompt templates for judging tutor responses."""

    @staticmethod
    def create_judgment_prompt(
        context: str,
        student_message: str,
        tutor_response: str,
        dimensions: List[Dict]
    ) -> str:
        """Create prompt for judging a tutor response.

        Args:
            context: Conversation history
            student_message: Student's current message
            tutor_response: Tutor's response to evaluate
            dimensions: List of evaluation dimensions

        Returns:
            Formatted prompt string
        """
        dimension_descriptions = "\n".join([
            f"- **{d['name'].replace('_', ' ').title()}** (weight: {d['weight']}): {d['description']}"
            for d in dimensions
        ])

        prompt = f"""You are an expert evaluator of language tutoring responses. Your task is to judge the quality of a tutor's response to a student.

## Conversation Context

{context}

## Current Exchange

**Student:** {student_message}

**Tutor Response to Evaluate:** {tutor_response}

## Evaluation Task

Rate the tutor's response on the following dimensions (scale 1-10, where 1 is poor and 10 is excellent):

{dimension_descriptions}

## Instructions

For each dimension:
1. Provide a score from 1-10
2. Briefly explain your reasoning (1-2 sentences)

Then provide an overall weighted score based on the dimension weights.

## Output Format

Respond with a JSON object in this exact format:
```json
{{
  "dimensions": {{
    "engagement": {{"score": <1-10>, "reasoning": "<explanation>"}},
    "accuracy": {{"score": <1-10>, "reasoning": "<explanation>"}},
    "clarity": {{"score": <1-10>, "reasoning": "<explanation>"}},
    "personalization": {{"score": <1-10>, "reasoning": "<explanation>"}},
    "pedagogical_value": {{"score": <1-10>, "reasoning": "<explanation>"}}
  }},
  "overall": {{
    "score": <weighted average>,
    "reasoning": "<overall assessment>"
  }}
}}
```

**CRITICAL:** Your response must be ONLY the JSON object above. Do not include any other text, markdown formatting, or code blocks.
"""
        return prompt

    @staticmethod
    def create_batch_prompt(conversations: List[Dict], dimensions: List[Dict]) -> List[str]:
        """Create prompts for a batch of conversations.

        Args:
            conversations: List of conversation dicts
            dimensions: Evaluation dimensions

        Returns:
            List of prompts
        """
        return [
            JudgePrompts.create_judgment_prompt(
                context=conv['context'],
                student_message=conv['student_message'],
                tutor_response=conv['tutor_response'],
                dimensions=dimensions
            )
            for conv in conversations
        ]
