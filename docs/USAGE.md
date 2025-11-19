# Usage Guide

## Quick Reference
```python
# Load and score with trained reward model
from src.reward_model.inference import RewardModelScorer

scorer = RewardModelScorer("models/reward_model_final")

# Score a single response
score = scorer.score(
    context="Student: Teach me grammar",
    response="Let's start with the basics..."
)

# Score multiple responses
scores = scorer.score_batch(contexts, responses)
```

## Installation

### 1. Clone and Setup
```bash
git clone https://github.com/your-username/fully-fluent-reward-model.git
cd fully-fluent-reward-model

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Training a Reward Model

### Step 1: Prepare Data

Create `data/conversations.json` with your conversation data. See [DATA_FORMAT.md](DATA_FORMAT.md) for complete specification.

### Step 2: Run Pipeline
```bash
# Full automated pipeline
bash scripts/run_full_pipeline.sh

# Or step by step:
python scripts/01_validate_data.py
python scripts/02_judge_responses.py
python scripts/03_train_reward_model.py
python scripts/04_evaluate_reward_model.py
```

### Step 3: Review Results

Check the outputs:
- **Model**: `models/reward_model_final/`
- **Metrics**: `outputs/metrics/evaluation_metrics.json`
- **Plots**: `outputs/metrics/evaluation_plots.png`

## Using the Trained Model

### Basic Scoring
```python
from src.reward_model.inference import RewardModelScorer

scorer = RewardModelScorer("models/reward_model_final")

context = "Student: I want to learn present tense."
response = "Great! Present tense is used for habits and facts."

score = scorer.score(context=context, response=response)
print(f"Quality Score: {score:.2f}/10")
```

### Batch Scoring
```python
contexts = [
    "Student: Help with grammar",
    "Student: Explain pronunciation",
    "Student: I need essay help"
]

responses = [
    "Let's start with basic grammar rules...",
    "Pronunciation is very important...",
    "I'll help you with your essay structure..."
]

scores = scorer.score_batch(contexts, responses, batch_size=32)
```

### Best-of-N Sampling
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.reward_model.inference import RewardModelScorer

# Load models
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
scorer = RewardModelScorer("models/reward_model_final")

# Generate N candidates
context = "Student: Help me with grammar"
candidates = []
for _ in range(5):
    inputs = tokenizer(context, return_tensors="pt")
    output = base_model.generate(**inputs, max_new_tokens=100, do_sample=True)
    candidates.append(tokenizer.decode(output[0], skip_special_tokens=True))

# Score and select best
scores = scorer.score_batch([context] * len(candidates), candidates)
best_idx = scores.index(max(scores))
print(f"Best response: {candidates[best_idx]}")
```

## Advanced Usage

### Custom Configuration

Edit `config/reward_model_config.yaml`:
```yaml
model:
  base_model: "distilbert-base-uncased"  # Change architecture
  hidden_size: 768

training:
  num_epochs: 15  # More epochs
  batch_size: 32  # Larger batches
```

### Filtering by Metadata
```python
from src.data.loader import ConversationLoader

loader = ConversationLoader()
beginner_data = loader.filter_by_metadata("student_level", "beginner")
```

## Troubleshooting

### "CUDA out of memory"
Reduce batch size in config or use CPU:
```python
scorer = RewardModelScorer("models/reward_model_final", device="cpu")
```

### Low Correlation with Claude
If Pearson correlation < 0.7:
1. Collect more data (200+ examples minimum)
2. Check data quality
3. Try different architecture
4. Adjust training hyperparameters

### API Rate Limits
Increase delay in `config/judge_config.yaml`:
```yaml
batch:
  rate_limit_delay: 2
```

## Performance Optimization

### GPU Acceleration
```python
# Automatically uses GPU if available
scorer = RewardModelScorer("models/reward_model_final", device="auto")
```

## Resources

- **Documentation**: `docs/`
- **Examples**: `notebooks/`
- **Config files**: `config/`
- **Tests**: `tests/`
