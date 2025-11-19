# Fully Fluent Reward Model

**Distill Claude's judgment into a lightweight reward model for efficient best-of-N sampling and RLAIF.**

Train fast, local reward models that replicate Claude's evaluation of language tutor responses—enabling scalable response selection and reinforcement learning without API costs.

## 🎯 Purpose

This repo creates reward models that can:
- **Score tutor responses** on pedagogical quality (engagement, accuracy, personalization)
- **Enable best-of-N sampling** in your own code to select best responses
- **Be used in DPO training** 

## 🚀 Quick Start

### 1. Setup
```bash
# Clone repo
git clone https://github.com/your-username/fully-fluent-reward-model.git
cd fully-fluent-reward-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 2. Prepare Data

Place your conversation data in `data/conversations.json`:
```json
[
  {
    "context": "Full conversation history up to this point",
    "student_message": "Student's current message",
    "tutor_response": "Tutor's response to evaluate"
  }
]
```

**Required fields:**
- `context`: String with full conversation history
- `student_message`: String with student's current message
- `tutor_response`: String with tutor's response to judge

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for complete specification.

### 3. Run Pipeline
```bash
# Full pipeline (all steps)
bash scripts/run_full_pipeline.sh

# Or run step by step:
python scripts/01_validate_data.py
python scripts/02_judge_responses.py
python scripts/03_train_reward_model.py
python scripts/04_evaluate_reward_model.py
```

## 📊 Results

The pipeline produces:

- **Trained reward model**: `models/reward_model_final/`
- **Claude judgments**: `data/judgments/claude_judgments.json`
- **Performance metrics**: `outputs/metrics/`
- **Training logs**: `outputs/logs/`
- **Evaluation report**: Shows correlation between reward model and Claude scores

## 🔧 Using the Reward Model

### In Your Own Code
```python
from src.reward_model.inference import RewardModelScorer

# Load trained model
scorer = RewardModelScorer("models/reward_model_final")

# Score a single response
score = scorer.score(
    context="Student: I want to learn present tense.",
    response="Great! Let me explain present tense. It's used for habits..."
)
print(f"Quality Score: {score:.2f}")

# Score multiple responses (batched for efficiency)
scores = scorer.score_batch(
    contexts=[context1, context2, context3],
    responses=[response1, response2, response3]
)
```

### Best-of-N Sampling Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.reward_model.inference import RewardModelScorer

# Load your base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Load reward model
scorer = RewardModelScorer("models/reward_model_final")

# Generate N candidates
context = "Student: Help me with grammar"
candidates = []
for _ in range(5):
    inputs = tokenizer(context, return_tensors="pt")
    output = base_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )
    candidates.append(tokenizer.decode(output[0], skip_special_tokens=True))

# Score all candidates
scores = scorer.score_batch(
    contexts=[context] * len(candidates),
    responses=candidates
)

# Select best
best_idx = scores.index(max(scores))
best_response = candidates[best_idx]
print(f"Selected response with score {scores[best_idx]:.2f}")
```

See `docs/USAGE.md` for more detailed examples.

## 📚 Documentation

- [DATA_FORMAT.md](docs/DATA_FORMAT.md) - Required input data format
- [JUDGING_CRITERIA.md](docs/JUDGING_CRITERIA.md) - Evaluation dimensions explained
- [USAGE.md](docs/USAGE.md) - Code examples and integration patterns

## 🧪 Development

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Formatting
```bash
# Format code
black src/ scripts/ tests/

# Sort imports
isort src/ scripts/ tests/

# Lint
flake8 src/ scripts/ tests/
```

## 🔒 Privacy & Data


All conversation files are gitignored to protect privacy:
- `data/conversations.json` 
- `data/judgments/*` 
- `models/*`

Only example files and code are version controlled.


## 🛠️ Configuration

Customize behavior via YAML configs in `config/`:

### `judge_config.yaml`
- Evaluation dimensions and weights
- Claude model settings
- Prompt templates

### `reward_model_config.yaml`
- Model architecture
- Training hyperparameters
- Data split ratios


## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.
