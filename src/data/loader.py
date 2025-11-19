"""Load and manage conversation data."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)


class ConversationLoader:
    """Load and manage conversation data for reward model training."""

    def __init__(self, data_path: str = "data/conversations.json"):
        """Initialize loader.

        Args:
            data_path: Path to conversations JSON file
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                f"Please create this file with your conversation data.\n"
                f"See data/conversations.example.json for format."
            )

    def load(self) -> List[Dict]:
        """Load conversations from JSON.

        Returns:
            List of conversation dictionaries

        Raises:
            ValueError: If data format is invalid
        """
        logger.info(f"Loading conversations from {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Data file must contain a JSON array")

        # Validate required fields
        required_fields = ['context', 'student_message', 'tutor_response']
        for i, item in enumerate(data):
            missing = [f for f in required_fields if f not in item]
            if missing:
                raise ValueError(
                    f"Item {i} missing required fields: {missing}\n"
                    f"Required fields: {required_fields}"
                )

        logger.info(f"Loaded {len(data)} conversation turns")
        return data

    def load_as_dataset(self) -> Dataset:
        """Load as HuggingFace Dataset for easy manipulation.

        Returns:
            HuggingFace Dataset object
        """
        data = self.load()
        return Dataset.from_list(data)

    def get_sample(self, n: int = 5) -> List[Dict]:
        """Get a sample of conversations for inspection.

        Args:
            n: Number of samples to return

        Returns:
            List of sampled conversations
        """
        data = self.load()
        return data[:min(n, len(data))]

    def filter_by_metadata(
        self,
        key: str,
        value: str
    ) -> List[Dict]:
        """Filter conversations by metadata field.

        Args:
            key: Metadata key to filter on (e.g., 'student_level')
            value: Value to match (e.g., 'beginner')

        Returns:
            Filtered list of conversations
        """
        data = self.load()
        filtered = [
            item for item in data
            if item.get('metadata', {}).get(key) == value
        ]
        logger.info(
            f"Filtered {len(filtered)} items where {key}={value}"
        )
        return filtered

    def get_statistics(self) -> Dict:
        """Get basic statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        data = self.load()

        stats = {
            'total_turns': len(data),
            'unique_conversations': len(set(
                item.get('conversation_id', f'unknown_{i}')
                for i, item in enumerate(data)
            )),
            'avg_context_length': sum(
                len(item['context']) for item in data
            ) / len(data) if data else 0,
            'avg_response_length': sum(
                len(item['tutor_response']) for item in data
            ) / len(data) if data else 0,
        }

        # Metadata statistics
        if data and 'metadata' in data[0]:
            # Get all metadata keys
            all_keys = set()
            for item in data:
                if 'metadata' in item:
                    all_keys.update(item['metadata'].keys())

            stats['metadata_fields'] = list(all_keys)

            # Count values for each key
            for key in all_keys:
                values = [
                    item.get('metadata', {}).get(key)
                    for item in data
                    if 'metadata' in item and key in item['metadata']
                ]
                unique_values = set(values)
                stats[f'unique_{key}'] = len(unique_values)

        return stats
