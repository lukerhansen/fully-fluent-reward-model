# Data Format Specification

## Overview

This document specifies the required format for conversation data used to train reward models.

## File Location

Place your conversation data at: `data/conversations.json`

## Format

The file must be a JSON array containing conversation turn objects.

## Required Fields

Each conversation turn must have these fields:

### `context` (string)
Full conversation history up to this point, including both tutor and student messages.

**Example:**
```
"Tutor: Hello! How can I help you today?\nStudent: I want to learn present tense."
```

### `student_message` (string)
The student's current message in this turn.

**Example:**
```
"I want to learn present tense."
```

### `tutor_response` (string)
The tutor's response that will be evaluated by the reward model.

**Example:**
```
"Great! The present tense is used for habits and facts. For example: 'I eat breakfast every day.'"
```

## Optional Fields

### `conversation_id` (string)
Unique identifier for the conversation session.

**Example:** `"conv_001"`, `"session_abc123"`

### `turn_index` (integer)
Position of this turn within the conversation (0-indexed or 1-indexed).

**Example:** `2`

### `metadata` (object)
Additional information about the conversation or student.

**Recommended fields:**
- `timestamp`: ISO datetime string
- `student_level`: `"beginner"`, `"intermediate"`, or `"advanced"`
- `student_age`: `"child"`, `"teen"`, or `"adult"`
- `session_id`: Session identifier
- `topic`: Main topic being discussed

## Data Size Recommendations

### Minimum (for testing/demo)
- **50-100 conversation turns**
- Allows basic pipeline testing
- Not suitable for production

### Recommended (meaningful results)
- **200-500 conversation turns**
- Provides reasonable reward model quality
- Good for initial deployment

### Ideal (production quality)
- **1000+ conversation turns**
- High-quality reward model
- Better generalization
- More reliable scores

## Validation

Validate your data before running the pipeline:
```bash
python scripts/01_validate_data.py
```

## Privacy Considerations

**IMPORTANT:** This file is gitignored by default to protect privacy.

Before adding your data:
1. **Remove all PII** (names, emails, phone numbers, addresses)
2. **Anonymize student identifiers**
3. **Remove sensitive metadata** (IP addresses, user IDs)
4. **Consider synthetic data** for sensitive domains

## Troubleshooting

### "Missing required field" Error
- Check that all three required fields are present: `context`, `student_message`, `tutor_response`
- Ensure field names are exactly as specified (case-sensitive)

### "Field must be string" Error
- All three required fields must be strings
- Convert numbers or lists to strings if needed

### "Dataset is empty" Warning
- File contains empty array `[]`
- Make sure you have actual conversation data

### "Too few samples" Warning
- You have fewer than 50 conversation turns
- Consider collecting more data for better results
