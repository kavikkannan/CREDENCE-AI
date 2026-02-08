# Phase 3: Source Account Analysis Pipeline

## Overview
This phase evaluates credibility based on account behavior and source reliability using rule-based heuristics.

## Inputs
- `account.account_age_days`: Integer (age of account in days)
- `account.verified`: Boolean (whether account is verified)
- `account.historical_post_count`: Integer (number of historical posts)
- `urls`: Array of strings (URLs in the post)

### Optional Account Inputs
- `account.name`: String (account display name, e.g., "BBC")
- `account.screen_name`: String (account handle, e.g., "BBC")
- `account.description`: String (account description)
- `account.followers_count`: Integer (number of followers)

### Optional Inputs
- `known_domain_list`: Array of strings (trusted domains)
- `blacklisted_sources`: Array of strings (blacklisted domains)

## Outputs (source_signals schema)
- `account_trust_score`: Float (0-1) - Trustworthiness based on account characteristics
- `source_reliability_score`: Float (0-1) - Reliability of linked domains
- `behavioral_risk_flag`: Boolean - Indicates if risky behavior patterns detected

## Scoring Methodology

### Account Trust Score (0-1)
Computed from six factors:
1. **Account Age** (0-0.4 points)
   - 365+ days: 0.4
   - 180-364 days: 0.3
   - 90-179 days: 0.2
   - 30-89 days: 0.1
   - <30 days: 0.05 * (days/30)

2. **Verification Status** (0-0.3 points)
   - Verified: +0.3
   - Unverified: 0

3. **Historical Post Count** (0-0.2 points)
   - 1000+ posts: 0.2
   - 500-999 posts: 0.15
   - 100-499 posts: 0.1
   - 50-99 posts: 0.08
   - 10-49 posts: 0.05
   - 1-9 posts: 0.03

4. **Known Trusted Sources** (0-0.1 points)
   - If account name/screen_name matches known trusted sources (BBC, Reuters, AP, etc.): +0.1

5. **Followers Count** (0-0.05 points)
   - 1M+ followers: +0.05
   - 100K-1M followers: +0.03
   - 10K-100K followers: +0.02
   - 1K-10K followers: +0.01

6. **Description Quality** (0-0.05 points)
   - Professional descriptions with keywords (news, media, broadcaster, etc.): +0.05

### Source Reliability Score (0-1)
Evaluated based on:
- Blacklisted domains: 0.0
- Known trusted domains: 1.0
- TLD heuristics (.edu, .gov, .org): 0.7
- URL shorteners: 0.3
- News/media domains: +0.2 (max 0.8)
- Social media domains: 0.6
- Default: 0.5

### Behavioral Risk Flag
Detects risky patterns:
- Very new account (<30 days) with high activity (>100 posts)
- Account with no historical posts
- Very new account (<7 days)
- Unverified account with suspicious domain patterns
- Multiple URL shorteners (≥2)
- Unusually high posting rate (>10 posts/day)

## Usage

### Standalone execution
```bash
python source_scoring.py input.json source_outputs.json
```

### Programmatic usage
```python
from source_scoring import SourceAccountAnalyzer

# Initialize analyzer
analyzer = SourceAccountAnalyzer(
    known_domain_list=['reuters.com', 'bbc.com'],
    blacklisted_sources=['spam.com']
)

# Process single post
result = analyzer.process_post(
    post_id="post_123",
    account_age_days=365,
    verified=True,
    historical_post_count=500,
    urls=["https://www.reuters.com/article"]
)

# Process batch
posts = [
    {
        "post_id": "1",
        "account": {
            "account_age_days": 730,
            "verified": True,
            "historical_post_count": 1500
        },
        "urls": ["https://example.com"]
    }
]
analyzer.process_batch(posts, "source_outputs.json")
```

## Constraints
- Uses only rule-based scoring (no machine learning)
- All scores normalized between 0 and 1
- Outputs stored keyed by `post_id`
- Outputs match `source_signals` schema exactly
- Follows phase contracts strictly
