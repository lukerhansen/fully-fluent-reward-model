#!/usr/bin/env python3
"""Validate conversation data format."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import ConversationLoader
from src.data.validator import DataValidator
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__, "outputs/logs/validation.log")


def main():
    """Validate data and print statistics."""
    print("=" * 60)
    print("Data Validation")
    print("=" * 60)
    print()

    try:
        # Load data
        loader = ConversationLoader("data/conversations.json")
        data = loader.load()

        # Validate
        is_valid, report = DataValidator.validate_dataset(data)

        # Print results
        print(f"Total items: {report['total_items']}")
        print(f"Valid items: {report['valid_items']}")
        print(f"Invalid items: {report['invalid_items']}")
        print()

        if report['warnings']:
            print("⚠️  Warnings:")
            for warning in report['warnings']:
                print(f"  - {warning}")
            print()

        if not is_valid:
            print("❌ Validation FAILED")
            print("\nErrors:")
            for error in report['errors'][:5]:  # Show first 5
                print(f"  Item {error['index']}: {error['errors']}")

            if len(report['errors']) > 5:
                print(f"  ... and {len(report['errors']) - 5} more errors")

            sys.exit(1)

        # Get statistics
        stats = loader.get_statistics()
        print("✓ Validation PASSED")
        print("\nDataset Statistics:")
        print(f"  Total turns: {stats['total_turns']}")
        print(f"  Unique conversations: {stats['unique_conversations']}")
        print(f"  Avg context length: {stats['avg_context_length']:.0f} chars")
        print(f"  Avg response length: {stats['avg_response_length']:.0f} chars")

        if 'metadata_fields' in stats:
            print(f"  Metadata fields: {', '.join(stats['metadata_fields'])}")

        print()
        print("=" * 60)
        print("✓ Data is ready for judging!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease create data/conversations.json with your data.")
        print("See data/conversations.example.json for the required format.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
