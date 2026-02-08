# Phase 2: NLP Analysis Pipeline

## Overview
This phase extracts linguistic and emotional credibility signals from social media text content.

## Inputs
- `post_id`: Unique identifier for the post
- `text`: Main text content
- `hashtags`: Array of hashtags (optional)
- `urls`: Array of URLs (optional)

## Outputs (nlp_signals schema)
- `sentiment`: String (e.g., "POSITIVE", "NEGATIVE")
- `emotion`: String (e.g., "joy", "anger", "neutral")
- `clickbait`: Boolean
- `extracted_claim`: String (primary factual claim)
- `text_embedding_id`: String (hash-based identifier for embedding)

## Models Used
1. **Sentiment Analysis**: `distilbert-base-uncased-finetuned-sst-2-english`
2. **Emotion Detection**: `j-hartmann/emotion-english-distilroberta-base`
3. **Clickbait Detection**: `mrm8488/bert-tiny-finetuned-clickbait-news`
4. **Text Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`

## Usage

### Standalone execution
```bash
python nlp_pipeline.py input.json nlp_outputs.json
```

### Programmatic usage
```python
from nlp_pipeline import NLPAnalysisPipeline

# Initialize pipeline
pipeline = NLPAnalysisPipeline()

# Process single post
result = pipeline.process_post(
    post_id="post_123",
    text="Your text here",
    hashtags=["tag1", "tag2"],
    urls=["https://example.com"]
)

# Process batch
posts = [
    {"post_id": "1", "text": "...", "hashtags": [], "urls": []},
    {"post_id": "2", "text": "...", "hashtags": [], "urls": []}
]
pipeline.process_batch(posts, "nlp_outputs.json")
```

## Constraints
- Uses only pretrained HuggingFace models (no training)
- All outputs stored keyed by `post_id`
- Outputs match `nlp_signals` schema exactly
- Follows phase contracts strictly
