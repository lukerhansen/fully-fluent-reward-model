#!/bin/bash
# Run complete reward model training pipeline

set -e  # Exit on error

echo "======================================"
echo "Fully Fluent Reward Model Pipeline"
echo "======================================"
echo ""

# Check data file exists
if [ ! -f "data/conversations.json" ]; then
    echo "❌ ERROR: data/conversations.json not found!"
    echo ""
    echo "Please create this file with your conversation data."
    echo "See data/conversations.example.json for the required format."
    echo "See docs/DATA_FORMAT.md for complete specification."
    exit 1
fi

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ERROR: ANTHROPIC_API_KEY not set!"
    echo ""
    echo "Please set your Anthropic API key:"
    echo "  export ANTHROPIC_API_KEY='your-key-here'"
    echo ""
    echo "Or add it to your .env file."
    exit 1
fi

echo "✓ Data file found"
echo "✓ API key configured"
echo ""

# Create output directories
mkdir -p outputs/logs outputs/metrics outputs/figures
mkdir -p data/judgments models

# Step 1: Validate data
echo "======================================"
echo "Step 1/4: Validating Data"
echo "======================================"
python scripts/01_validate_data.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Data validation failed. Please fix the issues and try again."
    exit 1
fi
echo ""

# Step 2: Judge with Claude
echo "======================================"
echo "Step 2/4: Judging with Claude"
echo "======================================"
python scripts/02_judge_responses.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Judging failed. Check the logs for details."
    exit 1
fi
echo ""

# Step 3: Train reward model
echo "======================================"
echo "Step 3/4: Training Reward Model"
echo "======================================"
python scripts/03_train_reward_model.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed. Check the logs for details."
    exit 1
fi
echo ""

# Step 4: Evaluate
echo "======================================"
echo "Step 4/4: Evaluating Model"
echo "======================================"
python scripts/04_evaluate_reward_model.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Evaluation failed. Check the logs for details."
    exit 1
fi
echo ""

echo "======================================"
echo "✓ Pipeline Complete!"
echo "======================================"
echo ""
echo "Results:"
echo "  📁 Trained model: models/reward_model_final/"
echo "  📁 Judgments: data/judgments/claude_judgments.json"
echo "  📁 Metrics: outputs/metrics/"
echo "  📁 Logs: outputs/logs/"
echo ""
echo "Next steps:"
echo "  1. Review evaluation metrics in outputs/metrics/"
echo "  2. Check notebooks/ for analysis examples"
echo "  3. Use the model:"
echo "     from src.reward_model.inference import RewardModelScorer"
echo "     scorer = RewardModelScorer('models/reward_model_final')"
echo ""
echo "  4. Or copy to DPO trainer:"
echo "     cp -r models/reward_model_final ../fully-fluent-dpo-trainer/models/reward/"
echo "======================================"
