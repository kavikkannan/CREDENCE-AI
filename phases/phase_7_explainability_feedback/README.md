# Phase 7: Explainability and Feedback Pipeline

## Overview
This phase generates human-readable explanations from final decisions and captures optional user feedback for evaluation and learning.

## Inputs
- `final_decision`: Dictionary from phase 6 with:
  - `final_credibility_score`: Float (0-1)
  - `agent_agreement_level`: Float (0-1)
  - `reasoning_trace`: Array of strings

### Optional Inputs
- `user_feedback`: String - User feedback ("true", "false", or "uncertain")

## Outputs (user_facing_output schema)
- `credibility_score`: Float (0-1) - Formatted credibility score (rounded to 2 decimals)
- `warning_label`: String - Warning label based on credibility:
  - "Credible" (score >= 0.7)
  - "Caution Advised" (0.5 <= score < 0.7)
  - "Low Credibility" (0.3 <= score < 0.5)
  - "High Risk - Verify Information" (score < 0.3)
- `explanation`: Array of strings - Human-readable explanation strings

### Optional Outputs
- `feedback_record`: Object - Feedback record (if user_feedback provided):
  - `post_id`: String
  - `timestamp`: ISO format timestamp
  - `user_feedback`: String
  - `system_prediction`: Object with credibility_score and warning_label
  - `final_decision`: Object with final_credibility_score and agent_agreement_level

## Processing

### Explanation Generation
Converts technical reasoning traces into human-readable explanations:
1. **Main Assessment**: Credibility level with confidence percentage
2. **Agent Agreement**: Level of agreement between analysis systems
3. **Key Insights**: Extracted from reasoning trace:
   - **Text Analysis**: Sentiment, clickbait, emotional language
   - **Image Analysis**: Tampering detection, AI-generated probability
   - **Source Analysis**: Account trust, behavioral patterns

### Warning Label Generation
Generates appropriate warning labels based on credibility score:
- High credibility (≥0.7): "Credible"
- Moderate credibility (0.5-0.7): "Caution Advised"
- Low credibility (0.3-0.5): "Low Credibility"
- Very low credibility (<0.3): "High Risk - Verify Information"

### Feedback Logging
Captures and logs user feedback for:
- System evaluation
- Performance metrics
- Learning and improvement (phase 8)

## Usage

### Standalone execution
```bash
python explainability.py input.json final_output.json [feedback_log.json]
```

### Programmatic usage
```python
from explainability import ExplainabilityFeedbackPipeline

# Initialize pipeline
pipeline = ExplainabilityFeedbackPipeline(feedback_file="feedback_log.json")

# Process single post
result = pipeline.process_post(
    post_id="post_123",
    final_decision={
        "final_credibility_score": 0.75,
        "agent_agreement_level": 0.8,
        "reasoning_trace": [
            "TextAgent: Positive sentiment detected (+0.1)",
            "SourceAgent: Account trust score: 0.900"
        ]
    },
    user_feedback="true"  # Optional
)

# Process batch
posts = [
    {
        "post_id": "1",
        "final_decision": {
            "final_credibility_score": 0.75,
            "agent_agreement_level": 0.8,
            "reasoning_trace": [...]
        },
        "user_feedback": "true"  # Optional
    }
]
pipeline.process_batch(posts, "final_output.json")
```

## Input Format
Each post should contain:
```json
{
  "post_id": "post_123",
  "final_decision": {
    "final_credibility_score": 0.75,
    "agent_agreement_level": 0.8,
    "reasoning_trace": [
      "TextAgent: Positive sentiment detected (+0.1)",
      "ImageAgent: No image tampering detected (+0.1)",
      "SourceAgent: Account trust score: 0.900"
    ]
  },
  "user_feedback": "true"  // Optional: "true", "false", or "uncertain"
}
```

## Explanation Format
Explanations are structured as:
1. **Main credibility assessment** with confidence percentage
2. **Agent agreement statement** (strong/moderate/some disagreement)
3. **Text Analysis insights** (sentiment, clickbait, emotions)
4. **Image Analysis insights** (tampering, AI-generation)
5. **Source Analysis insights** (trust, behavioral patterns)

## Feedback Logging
Feedback is logged to a JSON file with:
- Post ID and timestamp
- User feedback value
- System prediction
- Final decision details

This data is used in phase 8 for evaluation and learning.

## Constraints
- Explanations must be human-readable (no technical jargon)
- Do not expose raw model logits or internal scores
- All outputs stored keyed by `post_id`
- Outputs match `user_facing_output` schema exactly
- Follows phase contracts strictly
- User feedback is optional

## Design Principles
- **Human-Readable**: All explanations use plain language
- **Non-Technical**: No exposure of raw model outputs or logits
- **Informative**: Provides actionable insights
- **Traceable**: Links to reasoning from previous phases (in human terms)
