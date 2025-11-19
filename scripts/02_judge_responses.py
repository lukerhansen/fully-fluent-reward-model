#!/usr/bin/env python3
"""Judge tutor responses using Claude."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import ConversationLoader
from src.judge.claude_judge import ClaudeJudge
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__, "outputs/logs/judging.log")


def main(args):
    """Run Claude judging pipeline."""
    print("=" * 60)
    print("Claude Judge")
    print("=" * 60)
    print()

    # Load data
    print(f"Loading data from {args.input_file}...")
    loader = ConversationLoader(data_path=args.input_file)
    conversations = loader.load()
    print(f"✓ Loaded {len(conversations)} conversation turns")
    print()

    # Sample if requested
    if args.max_samples:
        conversations = conversations[:args.max_samples]
        print(f"Using {len(conversations)} samples for judging")
        print()

    # Initialize judge
    print("Initializing Claude judge...")
    judge = ClaudeJudge(config_path=args.config)
    print("✓ Judge initialized")
    print()

    # Judge responses
    print(f"Judging {len(conversations)} responses...")
    print("This may take a while...")
    print()

    judgments = judge.batch_judge(
        contexts=conversations,
        save_path=args.output_file
    )

    # Summary statistics
    print()
    print("=" * 60)
    print("Judging Complete!")
    print("=" * 60)
    print(f"  Total judgments: {len(judgments)}")
    print(f"  Saved to: {args.output_file}")

    if judgments:
        scores = [j['judgment']['overall']['score'] for j in judgments]
        print(f"\nScore Statistics:")
        print(f"  Mean: {sum(scores) / len(scores):.2f}")
        print(f"  Min: {min(scores):.1f}")
        print(f"  Max: {max(scores):.1f}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge tutor responses using Claude"
    )
    parser.add_argument(
        "--input-file",
        default="data/conversations.json",
        help="Path to conversations JSON file"
    )
    parser.add_argument(
        "--output-file",
        default="data/judgments/claude_judgments.json",
        help="Path to save judgments"
    )
    parser.add_argument(
        "--config",
        default="config/judge_config.yaml",
        help="Judge configuration file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to judge (for testing)"
    )

    args = parser.parse_args()
    main(args)
