# GenAI Social Media Credibility Analyzer - Complete Pipeline Overview

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Pipeline Phases](#pipeline-phases)
3. [Data Flow](#data-flow)
4. [Input Schema](#input-schema)
5. [Output Schema](#output-schema)
6. [Phase Details](#phase-details)
7. [Integration Points](#integration-points)

---

## System Architecture

The GenAI Social Media Credibility Analyzer is a **multi-phase, multi-agent pipeline** designed to assess the credibility of social media content using explainable AI techniques.

### Design Principles
- **Modular**: Each phase operates independently with clear contracts
- **Phase-driven**: Sequential processing with defined inputs/outputs
- **Explainable**: All decisions are traceable and human-readable
- **Research-oriented**: Suitable for academic research and prototyping

### Execution Mode
- **Local Prototype**: Runs entirely on local machine
- **No Cloud Dependencies**: All processing happens locally
- **Pretrained Models Only**: No model training required

---

## Pipeline Phases

The system consists of **8 sequential phases** that process social media posts through various analysis stages:

### Visual Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Data Ingestion                       │
│              Raw Social Media Data → Normalized Schema            │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│  PHASE 2  │ │  PHASE 3  │ │  PHASE 4  │
│   NLP     │ │  Source   │ │  Image    │
│ Analysis  │ │ Account   │ │ Analysis  │
└─────┬─────┘ └─────┬─────┘ └─────┬─────┘
      │             │             │
      │             │             │
      └─────────────┼─────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   PHASE 5:            │
        │ Misinformation        │
        │   Modeling            │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   PHASE 6:             │
        │ Agentic Reasoning      │
        │  (Multi-Agent)         │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   PHASE 7:             │
        │ Explainability &       │
        │    Feedback           │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   PHASE 8:             │
        │ Evaluation & Learning  │
        │   (Final Output)       │
        └───────────────────────┘
```

### Phase Dependencies

- **Phase 1** → Phases 2, 3, 4 (parallel processing)
- **Phases 2, 3, 4** → Phase 5 (signal fusion)
- **Phase 5** → Phase 6 (misinformation assessment)
- **Phase 6** → Phase 7 (final decision)
- **Phase 7** → Phase 8 (user feedback)

---

## Data Flow

### Input → Phase 1
**Raw Social Media Data** (platform-specific format)
- Twitter posts, Facebook posts, etc.
- Platform-specific fields and structures

### Phase 1 → Phases 2, 3, 4
**Normalized Post Object** (unified schema)
- Standardized format across all platforms
- Contains: post_id, text, image_path, urls, hashtags, account info, etc.

### Phases 2, 3, 4 → Phase 5
**Individual Signal Sets**:
- **Phase 2 Output**: NLP signals (sentiment, emotion, clickbait, claim, embedding)
- **Phase 3 Output**: Source signals (account trust, source reliability, behavioral risk)
- **Phase 4 Output**: Image signals (OCR text, tampering, AI-generated probability)

### Phase 5 → Phase 6
**Misinformation Assessment**:
- Content credibility score (0-1)
- Risk category (low/medium/high)

### Phase 6 → Phase 7
**Final Decision**:
- Final credibility score (0-1)
- Agent agreement level (0-1)
- Reasoning trace (array of strings)

### Phase 7 → Phase 8
**User-Facing Output**:
- Credibility score (formatted)
- Warning label
- Human-readable explanation

### Phase 8
**Evaluation Metrics**:
- Accuracy, precision, recall
- Error analysis
- Improvement opportunities

---

## Input Schema

### Global Input Schema (Phase 1 Output)

```json
{
  "post_id": "string",
  "platform": "string",
  "text": "string",
  "image_path": "string | null",
  "urls": ["array", "of", "strings"],
  "hashtags": ["array", "of", "strings"],
  "likes": "integer",
  "retweets": "integer",
  "timestamp": "YYYY-MM-DD",
  "account": {
    "account_id": "string",
    "account_age_days": "integer",
    "verified": "boolean",
    "historical_post_count": "integer",
    "name": "string (optional)",
    "screen_name": "string (optional)",
    "description": "string (optional)",
    "followers_count": "integer (optional)",
    "friends_count": "integer (optional)",
    "profile_image_url": "string (optional)"
  }
}
```

### Example Input

```json
{
  "post_id": "post_001",
  "platform": "twitter",
  "text": "Breaking: Scientists discover new breakthrough!",
  "image_path": "data/media/image/example.jpg",
  "urls": ["https://example.com/article"],
  "hashtags": ["science", "breakthrough"],
  "likes": 2158,
  "retweets": 231,
  "timestamp": "2024-11-14",
  "account": {
    "account_id": "acc_19701628",
    "account_age_days": 6219,
    "verified": true,
    "historical_post_count": 38483,
    "name": "BBC",
    "screen_name": "BBC",
    "description": "The BBC is the world's leading public service broadcaster",
    "followers_count": 2387759
  }
}
```

---

## Output Schema

### Final Output (Phase 7)

```json
{
  "post_id": {
    "user_facing_output": {
      "credibility_score": 0.75,
      "warning_label": "Credible",
      "explanation": [
        "This content appears to be credible (confidence: 75%).",
        "Our analysis systems are in strong agreement about this assessment.",
        "Text Analysis: The text has a positive tone. The text conveys positive emotions.",
        "Source Analysis: The account appears trustworthy. The account shows normal behavioral patterns."
      ]
    },
    "feedback_record": {
      "post_id": "post_id",
      "timestamp": "2024-01-15T10:30:00",
      "user_feedback": "true",
      "system_prediction": {
        "credibility_score": 0.75,
        "warning_label": "Credible"
      }
    }
  }
}
```

---

## Phase Details

### Phase 1: Data Ingestion & Normalization

**Purpose**: Convert raw or platform-specific data into a unified schema

**Inputs**:
- Raw social media data (platform-specific format)

**Processing**:
- Field extraction
- Schema mapping
- Format standardization
- Data validation

**Outputs**:
- Normalized post objects matching `global_input_schema`

**Passes To**: Phases 2, 3, 4

---

### Phase 2: NLP Analysis

**Purpose**: Extract linguistic and emotional credibility signals from text

**Inputs from Phase 1**:
- `post_id`
- `text`
- `hashtags`
- `urls`

**Models Used**:
- **Sentiment Analysis**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Emotion Detection**: `j-hartmann/emotion-english-distilroberta-base`
- **Clickbait Detection**: Rule-based (original model unavailable)
- **Text Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`

**Processing**:
1. **Sentiment Analysis**: Classifies text as POSITIVE, NEGATIVE, or NEUTRAL
2. **Emotion Detection**: Identifies emotions (joy, anger, fear, sadness, etc.)
3. **Clickbait Detection**: Uses rule-based heuristics to detect clickbait patterns
4. **Claim Extraction**: Extracts primary factual claim from text
5. **Text Embedding**: Generates embedding vector and creates hash-based ID

**Outputs** (`nlp_signals`):
```json
{
  "sentiment": "POSITIVE",
  "emotion": "joy",
  "clickbait": false,
  "extracted_claim": "Scientists discover new breakthrough in renewable energy",
  "text_embedding_id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
}
```

**Passes To**: Phases 5, 6

---

### Phase 3: Source Account Analysis

**Purpose**: Assess credibility based on account behavior and source reliability

**Inputs from Phase 1**:
- `account.account_age_days`
- `account.verified`
- `account.historical_post_count`
- `account.name` (optional)
- `account.screen_name` (optional)
- `account.description` (optional)
- `account.followers_count` (optional)
- `urls`

**Processing**:
1. **Account Trust Scoring**: Computes trust score based on:
   - Account age (0-0.4 points)
   - Verification status (0-0.3 points)
   - Historical post count (0-0.2 points)
   - Known trusted sources (0-0.1 points) - NEW
   - Followers count (0-0.05 points) - NEW
   - Description quality (0-0.05 points) - NEW

2. **Domain Reliability Scoring**: Evaluates URLs:
   - Blacklisted domains → 0.0
   - Known trusted domains → 1.0
   - TLD heuristics (.edu, .gov, .org) → 0.7
   - URL shorteners → 0.3
   - News/media domains → +0.2
   - Social media domains → 0.6

3. **Behavioral Heuristics**: Detects risky patterns:
   - Very new account with high activity
   - No historical posts
   - Suspicious domain patterns
   - Multiple URL shorteners
   - Unusually high posting rate

**Outputs** (`source_signals`):
```json
{
  "account_trust_score": 0.95,
  "source_reliability_score": 0.8,
  "behavioral_risk_flag": false
}
```

**Passes To**: Phases 5, 6

---

### Phase 4: Image Analysis

**Purpose**: Analyze visual content for manipulation and hidden textual cues

**Inputs from Phase 1**:
- `image_path` (can be null)

**Tools Used**:
- OpenCV (image processing)
- Tesseract OCR (text extraction)
- scikit-image (image analysis)

**Processing**:
1. **OCR Text Extraction**: Extracts text from images using Tesseract
2. **Image Tampering Detection**: Uses Error Level Analysis (ELA) to detect:
   - Compression inconsistencies
   - Areas of modification
   - Manipulation indicators
3. **AI-Generated Probability**: Rule-based heuristics:
   - Smoothness analysis (Laplacian variance)
   - Symmetry detection
   - Color distribution entropy

**Outputs** (`image_signals`):
```json
{
  "ocr_text": "Extracted text from image",
  "image_tampered": false,
  "ai_generated_probability": 0.15
}
```

**Note**: If `image_path` is null, returns default values (empty OCR, no tampering, 0.0 AI probability)

**Passes To**: Phases 2, 5, 6

---

### Phase 5: Misinformation Modeling

**Purpose**: Fuse multimodal signals to estimate content-level misinformation risk

**Inputs**:
- From Phase 2: `nlp_signals`
- From Phase 3: `source_signals`
- From Phase 4: `image_signals` (optional)
- Original `text` (for fake news classification)

**Models Used**:
- **Fake News Classification**: Uses sentiment model as proxy (original model requires 500MB+ disk space)

**Processing**:
1. **Fake News Classification**: 
   - Uses sentiment model to estimate fake news probability
   - Falls back to rule-based detection if model unavailable
   - Rule-based checks: capitalization, clickbait phrases, punctuation, URL patterns

2. **Heuristic Rule Fusion**:
   - **NLP Risk** (30-40% weight):
     - Clickbait: +0.3 risk
     - Negative sentiment: +0.1 risk
     - Manipulative emotions: +0.2 risk
   - **Source Risk** (40-60% weight):
     - Low account trust: (1 - trust) * 0.4
     - Low source reliability: (1 - reliability) * 0.4
     - Behavioral risk flag: +0.2
   - **Image Risk** (0-30% weight, if available):
     - Image tampered: +0.4
     - High AI-generated probability: ai_prob * 0.3

3. **Credibility Score Computation**:
   - Formula: `credibility = 1.0 - ((fake_news_prob + combined_risk) / 2.0)`
   - Normalized to 0-1 range

4. **Risk Categorization**:
   - Low: credibility_score >= 0.7
   - Medium: 0.4 <= credibility_score < 0.7
   - High: credibility_score < 0.4

**Outputs** (`misinformation_assessment`):
```json
{
  "content_credibility_score": 0.75,
  "risk_category": "low"
}
```

**Passes To**: Phase 6

---

### Phase 6: Agentic Reasoning

**Purpose**: Perform multi-agent reasoning and conflict resolution across all signals

**Inputs**:
- From Phase 2: `nlp_signals`
- From Phase 3: `source_signals`
- From Phase 4: `image_signals` (optional)
- From Phase 5: `misinformation_assessment`

**Agents**:
1. **TextAgent**: Evaluates text-based credibility
   - Analyzes sentiment, clickbait, emotion
   - Integrates Phase 5 assessment
   - Weight: 35%

2. **ImageAgent**: Evaluates image-based credibility
   - Detects tampering, AI-generation
   - Weight: 25%

3. **SourceAgent**: Evaluates source-based credibility
   - Account trust, source reliability, behavioral flags
   - Weight: 40%

4. **AggregatorAgent**: Combines agent outputs
   - Weighted aggregation
   - Computes agreement level
   - Generates combined reasoning trace

**Processing**:
1. **Independent Agent Evaluation**: Each agent evaluates independently
2. **Weighted Aggregation**: Combines scores with configurable weights
3. **Agreement Calculation**: Computes agreement level (1 - normalized std_dev)
4. **Reasoning Trace Generation**: Collects all reasoning steps

**Outputs** (`final_decision`):
```json
{
  "final_credibility_score": 0.825,
  "agent_agreement_level": 0.85,
  "reasoning_trace": [
    "TextAgent: Positive sentiment detected (+0.1)",
    "TextAgent: No clickbait patterns detected (+0.05)",
    "ImageAgent: No image tampering detected (+0.1)",
    "SourceAgent: Account trust score: 0.900 (impact: 0.160)",
    "AggregatorAgent: Weighted aggregation: 0.825"
  ]
}
```

**Passes To**: Phase 7

---

### Phase 7: Explainability & Feedback

**Purpose**: Generate human-readable explanations and capture user feedback

**Inputs from Phase 6**:
- `final_decision` (final_credibility_score, agent_agreement_level, reasoning_trace)

**Optional Inputs**:
- `user_feedback`: "true", "false", or "uncertain"

**Processing**:
1. **Explanation Generation**:
   - Converts technical reasoning to human-readable text
   - Extracts key insights from reasoning trace
   - Provides confidence percentages
   - No raw model logits exposed

2. **Warning Label Generation**:
   - "Credible" (score >= 0.7)
   - "Caution Advised" (0.5 <= score < 0.7)
   - "Low Credibility" (0.3 <= score < 0.5)
   - "High Risk - Verify Information" (score < 0.3)

3. **Feedback Logging** (if provided):
   - Captures user feedback
   - Stores with timestamp
   - Links to system prediction

**Outputs** (`user_facing_output`):
```json
{
  "credibility_score": 0.83,
  "warning_label": "Credible",
  "explanation": [
    "This content appears to be credible (confidence: 83%).",
    "Our analysis systems are in strong agreement about this assessment.",
    "Text Analysis: The text has a positive tone. The text conveys positive emotions.",
    "Source Analysis: The account appears trustworthy."
  ]
}
```

**Passes To**: Phase 8

---

### Phase 8: Evaluation & Learning

**Purpose**: Evaluate system performance and document results

**Inputs from Phase 7**:
- `feedback_record` (with user_feedback)
- `final_decision`

**Processing**:
1. **Performance Metrics Computation**:
   - Accuracy: (TP + TN) / Total
   - Precision: TP / (TP + FP)
   - Recall: TP / (TP + FN)
   - Confusion matrix (TP, FP, TN, FN)

2. **Error Analysis**:
   - False positive identification
   - False negative identification
   - Pattern identification
   - Systematic bias detection

3. **Improvement Opportunities**:
   - Threshold adjustments
   - Signal weight modifications
   - Pattern-specific recommendations

**Outputs** (`evaluation_metrics`):
```json
{
  "evaluation_metrics": {
    "accuracy": 0.875,
    "precision": 0.833,
    "recall": 0.909
  },
  "detailed_metrics": {
    "confusion_matrix": {
      "tp": 10,
      "fp": 2,
      "tn": 5,
      "fn": 1
    },
    "total_samples": 18,
    "evaluated_samples": 18
  },
  "error_analysis": {
    "false_positives": [...],
    "false_negatives": [...],
    "error_patterns": {...},
    "improvement_opportunities": [...]
  },
  "summary": "System shows strong overall performance..."
}
```

**Passes To**: None (final phase)

---

## Integration Points

### Frontend-Backend Integration

**Backend API** (`api_server.py`):
- FastAPI server on port 8000
- Endpoints:
  - `GET /api/dataset` - Load posts
  - `POST /api/analyze-single` - Analyze one post
  - `POST /api/analyze` - Analyze multiple posts
  - `GET /api/results/{post_id}` - Get saved results
  - `GET /health` - Health check

**Frontend** (`frontend/app/`):
- Next.js application on port 3000
- Calls backend API for all operations
- Displays real-time analysis results
- Shows pipeline progress

### Data Flow in Integration

```
Frontend (Next.js)
    ↓ HTTP Request
Backend API (FastAPI)
    ↓ Function Calls
Pipeline Phases (Python)
    ↓ JSON Response
Backend API
    ↓ HTTP Response
Frontend
    ↓ Display
User Interface
```

### File Structure

```
creditability-analyzer/
├── api_server.py              # FastAPI backend
├── run_pipeline.py            # Standalone pipeline runner
├── phases/
│   ├── phase_2_nlp_analysis/
│   ├── phase_3_source_account_analysis/
│   ├── phase_4_image_analysis/
│   ├── phase_5_misinformation_modeling/
│   ├── phase_6_agentic_reasoning/
│   ├── phase_7_explainability_feedback/
│   └── phase_8_evaluation_learning/
├── frontend/                  # Next.js frontend
│   └── app/
│       ├── page.tsx           # Main dashboard
│       └── api.ts            # API service
├── data/
│   ├── twitter_dataset.json  # Input dataset
│   └── media/image/          # Image files
├── outputs/                   # Pipeline outputs
└── docs/
    ├── Master_pipeline_context.json
    └── PIPELINE_OVERVIEW.md   # This document
```

---

## Key Features

### 1. Multimodal Analysis
- **Text**: Sentiment, emotion, clickbait, claims
- **Source**: Account trust, domain reliability, behavior
- **Image**: OCR, tampering, AI-generation

### 2. Multi-Agent Reasoning
- Independent agent evaluation
- Weighted aggregation
- Conflict resolution
- Agreement level calculation

### 3. Explainability
- Human-readable explanations
- Reasoning traces
- No raw model logits exposed
- Actionable insights

### 4. Evaluation & Learning
- Performance metrics
- Error analysis
- Improvement opportunities
- Feedback integration

---

## Constraints & Rules

### Global Rules
1. **Follow phase contracts strictly**: Each phase must match its defined input/output schema
2. **No cloud or deployment logic**: Everything runs locally
3. **Use pretrained models only**: No model training
4. **Prefer explainability over optimization**: Human-readable outputs prioritized
5. **Outputs must match defined schemas exactly**: Strict schema compliance

### Phase-Specific Rules
- **Phase 2**: Use HuggingFace pipelines, store outputs keyed by post_id
- **Phase 3**: Rule-based scoring only, normalize scores 0-1
- **Phase 4**: Skip if image_path is null, AI probability may be simulated
- **Phase 5**: Do not retrain classifiers, output risk category
- **Phase 6**: Agents as Python classes, no reinforcement learning, explainable decisions
- **Phase 7**: Human-readable explanations, no raw model logits
- **Phase 8**: Handle small datasets, qualitative + quantitative analysis

---

## Usage Examples

### Standalone Pipeline
```bash
python run_pipeline.py
```

### API Server
```bash
python api_server.py
```

### Frontend
```bash
cd frontend
npm run dev
```

### Analyze Single Post
```python
from phases.phase_2_nlp_analysis.nlp_pipeline import NLPAnalysisPipeline

pipeline = NLPAnalysisPipeline()
result = pipeline.process_post(
    post_id="post_001",
    text="Example text",
    hashtags=["tag1"],
    urls=["https://example.com"]
)
```

---

## Performance Considerations

### Model Loading
- Models are loaded once on startup (API server) or first use (standalone)
- First run downloads models (5-10 minutes)
- Subsequent runs use cached models

### Processing Time
- **Phase 2**: ~1-2 seconds per post (NLP models)
- **Phase 3**: <0.1 seconds per post (rule-based)
- **Phase 4**: ~2-5 seconds per post (image processing, if image exists)
- **Phase 5**: ~1 second per post (fake news classification)
- **Phase 6**: <0.1 seconds per post (rule-based aggregation)
- **Phase 7**: <0.1 seconds per post (text generation)
- **Phase 8**: <0.1 seconds per batch (metrics computation)

**Total**: ~5-10 seconds per post (with image), ~3-5 seconds (without image)

### Resource Requirements
- **Disk Space**: ~2GB for models
- **RAM**: ~4GB recommended
- **CPU**: Any modern CPU (GPU optional, not required)

---

## Future Extensibility

### Planned Extensions
- Additional platforms (Facebook, Instagram, YouTube)
- Live processing capabilities
- Real-time alerts
- Horizontal scaling support
- Additional trusted source databases
- Enhanced image analysis models
- Custom agent weight configurations

### Extension Points
- Phase 1: Add new platform parsers
- Phase 2: Add new NLP models
- Phase 3: Add new trusted source lists
- Phase 4: Add new image analysis techniques
- Phase 5: Add new fusion strategies
- Phase 6: Add new agents or modify weights
- Phase 7: Customize explanation templates
- Phase 8: Add new evaluation metrics

---

## Conclusion

The GenAI Social Media Credibility Analyzer provides a comprehensive, explainable approach to assessing social media content credibility. Through its 8-phase pipeline, it combines multimodal analysis, multi-agent reasoning, and human-readable explanations to help users make informed decisions about content trustworthiness.

All phases are designed to be modular, traceable, and research-oriented, making it suitable for academic research, prototyping, and local demonstrations.
