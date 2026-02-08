# Phase 6: Agentic Reasoning Pipeline

## Overview
This phase performs multi-agent reasoning and conflict resolution across all signals from previous phases using independent agent evaluations and weighted aggregation.

## Inputs
- `nlp_signals`: Dictionary from phase 2 (sentiment, emotion, clickbait, extracted_claim, text_embedding_id)
- `source_signals`: Dictionary from phase 3 (account_trust_score, source_reliability_score, behavioral_risk_flag)
- `image_signals`: Dictionary from phase 4 (optional) (ocr_text, image_tampered, ai_generated_probability)
- `misinformation_assessment`: Dictionary from phase 5 (content_credibility_score, risk_category)

### Optional Inputs
- `agent_weights`: Dictionary mapping agent names to weights (default: TextAgent=0.35, ImageAgent=0.25, SourceAgent=0.40)
- `decision_rules`: Object with custom decision rules (optional)

## Outputs (final_decision schema)
- `final_credibility_score`: Float (0-1) - Final credibility score from weighted aggregation
- `agent_agreement_level`: Float (0-1) - Level of agreement between agents (1 = high agreement)
- `reasoning_trace`: Array of strings - Step-by-step reasoning from all agents

## Agents

### TextAgent
Evaluates credibility based on text/NLP signals:
- **Sentiment**: Positive (+0.1), Negative (-0.1), Neutral (0)
- **Clickbait**: Detected (-0.2), Not detected (+0.05)
- **Emotion**: Manipulative emotions (anger/fear/disgust) (-0.15), Positive emotions (+0.05)
- **Phase 5 Integration**: Averages with misinformation assessment credibility score

### ImageAgent
Evaluates credibility based on image signals:
- **Image Tampering**: Detected (-0.3), Not detected (+0.1)
- **AI-Generated Probability**: 
  - High (>0.7): -0.25
  - Moderate (0.4-0.7): -0.1
  - Low (<0.2): +0.05
- **OCR Text**: Logs if extracted (neutral impact)
- Returns neutral (0.5) if no image signals available

### SourceAgent
Evaluates credibility based on source/account signals:
- **Account Trust Score**: Scaled impact of (trust_score - 0.5) * 0.4
- **Source Reliability Score**: Scaled impact of (reliability_score - 0.5) * 0.4
- **Behavioral Risk Flag**: Detected (-0.2), Not detected (+0.05)

### AggregatorAgent
Combines outputs from all agents:
- **Weighted Aggregation**: Combines agent scores using configurable weights
- **Default Weights**: TextAgent (35%), ImageAgent (25%), SourceAgent (40%)
- **Agent Agreement Level**: Calculated as 1 - (std_dev / 0.5) of agent scores
- **Reasoning Trace**: Combines all agent reasoning steps

## Processing Flow

1. **Independent Agent Evaluation**: Each agent evaluates independently based on its signals
2. **Weighted Aggregation**: AggregatorAgent combines agent outputs with weighted average
3. **Agreement Calculation**: Computes agreement level based on score variance
4. **Reasoning Trace Generation**: Collects all reasoning steps from all agents

## Usage

### Standalone execution
```bash
python agents.py input.json agent_outputs.json
```

### Programmatic usage
```python
from agents import MultiAgentReasoningSystem

# Initialize system with custom weights (optional)
agent_weights = {
    "TextAgent": 0.35,
    "ImageAgent": 0.25,
    "SourceAgent": 0.40
}
system = MultiAgentReasoningSystem(agent_weights=agent_weights)

# Process single post
result = system.process_post(
    post_id="post_123",
    nlp_signals={
        "sentiment": "POSITIVE",
        "emotion": "joy",
        "clickbait": False,
        "extracted_claim": "Example claim",
        "text_embedding_id": "abc123"
    },
    source_signals={
        "account_trust_score": 0.8,
        "source_reliability_score": 0.7,
        "behavioral_risk_flag": False
    },
    image_signals={
        "ocr_text": "",
        "image_tampered": False,
        "ai_generated_probability": 0.1
    },
    misinformation_assessment={
        "content_credibility_score": 0.75,
        "risk_category": "low"
    }
)

# Process batch
posts = [
    {
        "post_id": "1",
        "nlp_signals": {...},
        "source_signals": {...},
        "image_signals": {...},  # Optional
        "misinformation_assessment": {...}
    }
]
system.process_batch(posts, "agent_outputs.json")
```

## Input Format
Each post should contain:
```json
{
  "post_id": "post_123",
  "nlp_signals": {
    "sentiment": "POSITIVE",
    "emotion": "joy",
    "clickbait": false,
    "extracted_claim": "Claim text",
    "text_embedding_id": "embedding_id"
  },
  "source_signals": {
    "account_trust_score": 0.8,
    "source_reliability_score": 0.7,
    "behavioral_risk_flag": false
  },
  "image_signals": {
    "ocr_text": "",
    "image_tampered": false,
    "ai_generated_probability": 0.1
  },
  "misinformation_assessment": {
    "content_credibility_score": 0.75,
    "risk_category": "low"
  }
}
```

## Agent Agreement Level
- **High Agreement (0.8-1.0)**: Agents agree on credibility assessment
- **Medium Agreement (0.5-0.8)**: Some disagreement between agents
- **Low Agreement (0.0-0.5)**: Significant disagreement, may indicate ambiguous content

## Constraints
- Agents implemented as Python classes (no reinforcement learning)
- Final decision is explainable through reasoning trace
- All outputs stored keyed by `post_id`
- Outputs match `final_decision` schema exactly
- Follows phase contracts strictly
- Handles optional image_signals gracefully

## Design Principles
- **Independence**: Each agent evaluates independently
- **Explainability**: All reasoning steps are traceable
- **Weighted Aggregation**: Configurable agent importance
- **Conflict Resolution**: Agreement level indicates consensus
