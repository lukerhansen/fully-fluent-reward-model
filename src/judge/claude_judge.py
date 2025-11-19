"""Claude-based judge for evaluating tutor responses."""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
import yaml
from anthropic import Anthropic
from tqdm import tqdm
import logging

from src.judge.prompts import JudgePrompts
from src.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


class ClaudeJudge:
    """Use Claude to judge tutor response quality."""

    def __init__(self, config_path: str = "config/judge_config.yaml"):
        """Initialize Claude judge.

        Args:
            config_path: Path to judge configuration file
        """
        self.config = self._load_config(config_path)
        self.client = self._initialize_client()
        self.prompts = JudgePrompts()

        logger.info(f"Initialized Claude judge with model: {self.config['api']['model']}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _initialize_client(self) -> Anthropic:
        """Initialize Anthropic API client.

        Returns:
            Anthropic client instance

        Raises:
            ValueError: If API key not found
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment.\n"
                "Please set it in your .env file or environment variables."
            )
        return Anthropic(api_key=api_key)

    def judge_single(
        self,
        context: str,
        student_message: str,
        tutor_response: str,
        retry_attempts: Optional[int] = None
    ) -> Dict:
        """Judge a single tutor response.

        Args:
            context: Conversation history
            student_message: Student's current message
            tutor_response: Tutor's response to judge
            retry_attempts: Number of retry attempts (uses config default if None)

        Returns:
            Judgment dictionary with scores and reasoning
        """
        prompt = self.prompts.create_judgment_prompt(
            context=context,
            student_message=student_message,
            tutor_response=tutor_response,
            dimensions=self.config['dimensions']
        )

        retry_attempts = retry_attempts or self.config['batch']['retry_attempts']

        for attempt in range(retry_attempts):
            try:
                response = self.client.messages.create(
                    model=self.config['api']['model'],
                    max_tokens=self.config['api']['max_tokens'],
                    temperature=self.config['api']['temperature'],
                    messages=[{"role": "user", "content": prompt}]
                )

                # Extract and parse JSON response
                response_text = response.content[0].text.strip()

                # Remove markdown code blocks if present
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()

                judgment = json.loads(response_text)

                # Validate judgment structure
                self._validate_judgment(judgment)

                return judgment

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse JSON (attempt {attempt + 1}/{retry_attempts}): {e}\n"
                    f"Response: {response_text[:200]}"
                )
                if attempt == retry_attempts - 1:
                    raise ValueError(f"Failed to get valid JSON after {retry_attempts} attempts")
                time.sleep(self.config['batch']['retry_delay'])

            except Exception as e:
                logger.error(f"Error judging response (attempt {attempt + 1}/{retry_attempts}): {e}")
                if attempt == retry_attempts - 1:
                    raise
                time.sleep(self.config['batch']['retry_delay'])

        raise RuntimeError("Failed to get judgment after all retry attempts")

    def _validate_judgment(self, judgment: Dict) -> None:
        """Validate judgment structure.

        Args:
            judgment: Judgment dictionary to validate

        Raises:
            ValueError: If judgment is invalid
        """
        required_keys = ['dimensions', 'overall']
        for key in required_keys:
            if key not in judgment:
                raise ValueError(f"Judgment missing required key: {key}")

        # Check dimensions
        dimension_names = [d['name'] for d in self.config['dimensions']]
        for dim in dimension_names:
            if dim not in judgment['dimensions']:
                raise ValueError(f"Judgment missing dimension: {dim}")

            dim_data = judgment['dimensions'][dim]
            if 'score' not in dim_data or 'reasoning' not in dim_data:
                raise ValueError(f"Dimension {dim} missing score or reasoning")

        # Check overall
        if 'score' not in judgment['overall']:
            raise ValueError("Overall judgment missing score")

    def batch_judge(
        self,
        contexts: List[Dict],
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """Judge multiple responses in batch.

        Args:
            contexts: List of conversation dictionaries
            save_path: Path to save judgments (uses config default if None)

        Returns:
            List of judgment dictionaries
        """
        save_path = save_path or self.config['output']['save_path']
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        judgments = []

        logger.info(f"Judging {len(contexts)} responses...")

        for i, conv in enumerate(tqdm(contexts, desc="Judging responses")):
            try:
                judgment = self.judge_single(
                    context=conv['context'],
                    student_message=conv['student_message'],
                    tutor_response=conv['tutor_response']
                )

                # Add metadata
                judgment_with_meta = {
                    'conversation_id': conv.get('conversation_id', f'conv_{i}'),
                    'turn_index': conv.get('turn_index', i),
                    'context': conv['context'],
                    'student_message': conv['student_message'],
                    'tutor_response': conv['tutor_response'],
                    'judgment': judgment,
                    'metadata': conv.get('metadata', {})
                }

                judgments.append(judgment_with_meta)

                # Rate limiting
                if i < len(contexts) - 1:  # Don't sleep after last item
                    time.sleep(self.config['batch']['rate_limit_delay'])

                # Save incrementally every 10 judgments
                if (i + 1) % 10 == 0:
                    self._save_judgments(judgments, save_path)
                    logger.info(f"Saved {len(judgments)} judgments so far...")

            except Exception as e:
                logger.error(f"Failed to judge conversation {i}: {e}")
                # Continue with next conversation
                continue

        # Final save
        self._save_judgments(judgments, save_path)
        logger.info(f"✓ Completed {len(judgments)} judgments, saved to {save_path}")

        return judgments

    def _save_judgments(self, judgments: List[Dict], save_path: str) -> None:
        """Save judgments to JSON file.

        Args:
            judgments: List of judgment dictionaries
            save_path: Path to save file
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            if self.config['output']['pretty_print']:
                json.dump(judgments, f, indent=2, ensure_ascii=False)
            else:
                json.dump(judgments, f, ensure_ascii=False)
