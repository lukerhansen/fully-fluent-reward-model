"""Batch processing utilities for judging."""

import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process judgments in parallel batches."""

    def __init__(self, judge, max_workers: int = 5):
        """Initialize batch processor.

        Args:
            judge: ClaudeJudge instance
            max_workers: Maximum parallel workers
        """
        self.judge = judge
        self.max_workers = max_workers

    def process_parallel(
        self,
        conversations: List[Dict],
        show_progress: bool = True
    ) -> List[Dict]:
        """Process conversations in parallel.

        Args:
            conversations: List of conversation dicts
            show_progress: Whether to show progress bar

        Returns:
            List of judgments
        """
        judgments = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_conv = {
                executor.submit(
                    self.judge.judge_single,
                    conv['context'],
                    conv['student_message'],
                    conv['tutor_response']
                ): conv
                for conv in conversations
            }

            # Collect results with progress bar
            iterator = as_completed(future_to_conv)
            if show_progress:
                iterator = tqdm(iterator, total=len(conversations), desc="Judging")

            for future in iterator:
                conv = future_to_conv[future]
                try:
                    judgment = future.result()
                    judgments.append({
                        'conversation_id': conv.get('conversation_id'),
                        'judgment': judgment,
                        'original': conv
                    })
                except Exception as e:
                    logger.error(f"Error processing conversation: {e}")
                    continue

        return judgments
