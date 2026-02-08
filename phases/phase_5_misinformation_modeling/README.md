# Phase 5: Misinformation Modeling Pipeline

## Overview
This phase combines multimodal signals from previous phases to estimate content-level misinformation risk.

## Inputs
- `nlp_signals`: Dictionary from phase 2 (sentiment, emotion, clickbait, extracted_claim, text_embedding_id)
- `source_signals`: Dictionary from phase 3 (account_trust_score, source_reliability_score, behavioral_risk_flag)
- `image_signals`: Dictionary from phase 4 (optional) (ocr_text, image_tampered, ai_generated_probability)
- `text`: String (optional, original text for fake news classification)

## Outputs (misinformation_assessment schema)
- `content_credibility_score`: Float (0-1) - Overall credibility score (1 = most credible)
- `risk_category`: String - Risk level: "low", "medium", or "high"

## Processing Pipeline

### 1. Fake News Classification
- Uses pretrained model: `hamzab/roberta-fake-news-classification`
- Classifies text content (uses `extracted_claim` from NLP signals or provided text)
- Returns probability that content is fake news (0-1)

### 2. Heuristic Rule Fusion
Combines signals from multiple phases:

**NLP Risk Signals:**
- Clickbait detected: +0.3 risk
- Negative sentiment: +0.1 risk
- High emotional content (anger/fear/disgust): +0.2 risk
- Other emotions (joy/surprise): +0.1 risk

**Source Risk Signals:**
- Low account trust: (1 - trust_score) * 0.4
- Low source reliability: (1 - reliability_score) * 0.4
- Behavioral risk flag: +0.2 risk

**Image Risk Signals (if available):**
- Image tampered: +0.4 risk
- High AI-generated probability: ai_prob * 0.3

**Combined Risk:**
- With image signals: NLP (30%) + Source (40%) + Image (30%)
- Without image signals: NLP (40%) + Source (60%)

### 3. Credibility Score Computation
- Combines fake news probability and combined risk
- Formula: `credibility = 1.0 - ((fake_news_prob + combined_risk) / 2.0)`
- Normalized to 0-1 range (1 = most credible)

### 4. Risk Categorization
- **Low**: credibility_score >= 0.7
- **Medium**: 0.4 <= credibility_score < 0.7
- **High**: credibility_score < 0.4

## Usage

### Standalone execution
```bash
python misinformation_model.py input.json misinformation_outputs.json
```

### Programmatic usage
```python
from misinformation_model import MisinformationModel

# Initialize model
model = MisinformationModel()

# Process single post
result = model.process_post(
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
    }
)

# Process batch
posts = [
    {
        "post_id": "1",
        "nlp_signals": {...},
        "source_signals": {...},
        "image_signals": {...}  # Optional
    }
]
model.process_batch(posts, "misinformation_outputs.json")
```

## Input Format
Each post should contain:
```json
{
  "post_id": "post_123",
  "text": "Optional original text",
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
  }
}
```

## Constraints
- Uses only pretrained models (no retraining)
- Outputs risk category as "low", "medium", or "high"
- All outputs stored keyed by `post_id`
- Outputs match `misinformation_assessment` schema exactly
- Follows phase contracts strictly
- Handles optional image_signals gracefully

## Model Information
- **Fake News Classifier**: `hamzab/roberta-fake-news-classification`
- Uses HuggingFace transformers pipeline
- Model is pretrained and not retrained
