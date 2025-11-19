# Judging Criteria

## Overview

This document explains the evaluation dimensions used by Claude to judge tutor responses.

## Evaluation Dimensions

The reward model learns to score responses on five key dimensions:

### 1. Engagement (Weight: 1.0)

**Question:** How well does the response encourage student participation and interest?

**High scores (8-10):**
- Uses questions to prompt interaction
- References student's interests or previous statements
- Creates curiosity or excitement about learning
- Invites the student to practice or try something

**Medium scores (4-7):**
- Some attempt at interaction
- Basic acknowledgment of student
- Provides information but limited engagement

**Low scores (1-3):**
- One-way communication (lecture mode)
- No questions or interaction prompts
- Ignores student context
- Dry, unengaging delivery

### 2. Accuracy (Weight: 1.5)

**Question:** Is the linguistic/grammatical information correct?

**High scores (8-10):**
- All information is factually correct
- No misleading statements
- Appropriate use of terminology
- Examples are accurate

**Low scores (1-3):**
- Contains errors or misconceptions
- Incorrect grammar rules
- Wrong examples
- Misleading information

**Note:** This dimension has the highest weight (1.5) because accuracy is critical in educational contexts.

### 3. Clarity (Weight: 1.2)

**Question:** Is the explanation clear and appropriate for the student's level?

**High scores (8-10):**
- Simple, understandable language
- Well-structured explanation
- Appropriate for student's level
- Breaks down complex concepts
- Good examples that illustrate the point

**Low scores (1-3):**
- Confusing or convoluted
- Too complex for student level
- Technical jargon without explanation
- No examples or poor examples

### 4. Personalization (Weight: 1.0)

**Question:** Does the response adapt to the student's context, interests, and level?

**High scores (8-10):**
- References student's previous messages
- Uses student's interests in examples
- Adapts language to student's level
- Acknowledges student's specific situation
- Tailored approach

**Low scores (1-3):**
- Completely generic response
- Ignores student context
- One-size-fits-all approach
- No adaptation to level

### 5. Pedagogical Value (Weight: 1.3)

**Question:** Does the response effectively teach and reinforce learning?

**High scores (8-10):**
- Scaffolds learning appropriately
- Provides practice opportunities
- Reinforces previous learning
- Builds on student knowledge
- Includes feedback mechanisms

**Low scores (1-3):**
- Just gives answers without teaching
- No learning progression
- Doesn't help student understand "why"
- No opportunity to practice

## Overall Score Calculation

The overall score is a **weighted average** of the dimension scores:
```
Overall = (
    engagement * 1.0 +
    accuracy * 1.5 +
    clarity * 1.2 +
    personalization * 1.0 +
    pedagogical_value * 1.3
) / 6.0
```

### Weight Rationale

- **Accuracy (1.5):** Highest weight because incorrect information is harmful
- **Pedagogical Value (1.3):** Essential for effective teaching
- **Clarity (1.2):** Important for student understanding
- **Engagement (1.0):** Valuable but not as critical as accuracy
- **Personalization (1.0):** Beneficial but not always necessary

## Configuration

Dimensions and weights can be customized in `config/judge_config.yaml`

## Using These Criteria

### When Training the Reward Model
The model learns to internalize these criteria by studying Claude's judgments across hundreds of examples.

### When Generating Training Data
Create diverse examples that span the full range of quality (1-10) across all dimensions.

### When Evaluating Results
Check if the reward model's scores align with these criteria.

## References

- Configuration: `config/judge_config.yaml`
- Prompt templates: `src/judge/prompts.py`
- Judge implementation: `src/judge/claude_judge.py`
