"""
Quick test script to test a single post through the pipeline.
This helps verify setup before running the full dataset.
"""

import json
import os
import sys
from pathlib import Path

# Add phases to path
sys.path.insert(0, str(Path(__file__).parent))

from phases.phase_2_nlp_analysis.nlp_pipeline import NLPAnalysisPipeline
from phases.phase_3_source_account_analysis.source_scoring import SourceAccountAnalyzer
from phases.phase_4_image_analysis.image_analysis import ImageAnalyzer

def test_single_post():
    """Test a single post through first few phases"""
    
    # Load one post from dataset
    with open('data/twitter_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take first post
    post = data[0].copy()
    
    # Map image path
    if post.get('image_path'):
        image_filename = post['image_path']
        image_path = os.path.join('data', 'media', 'image', image_filename)
        if os.path.exists(image_path):
            post['image_path'] = image_path
        else:
            post['image_path'] = None
    
    print("Testing with post:", post['post_id'])
    print("Text:", post['text'][:100] + "...")
    print("\n" + "="*60)
    
    # Test Phase 2
    print("Testing Phase 2: NLP Analysis...")
    try:
        nlp_pipeline = NLPAnalysisPipeline()
        nlp_result = nlp_pipeline.process_post(
            post['post_id'],
            post.get('text', ''),
            post.get('hashtags', []),
            post.get('urls', [])
        )
        print("✓ Phase 2 completed")
        print(f"  Sentiment: {nlp_result['sentiment']}")
        print(f"  Emotion: {nlp_result['emotion']}")
        print(f"  Clickbait: {nlp_result['clickbait']}")
    except Exception as e:
        print(f"✗ Phase 2 failed: {e}")
        return
    
    # Test Phase 3
    print("\nTesting Phase 3: Source Account Analysis...")
    try:
        source_analyzer = SourceAccountAnalyzer()
        source_result = source_analyzer.process_post(
            post['post_id'],
            post['account']['account_age_days'],
            post['account']['verified'],
            post['account']['historical_post_count'],
            post.get('urls', [])
        )
        print("✓ Phase 3 completed")
        print(f"  Account Trust: {source_result['account_trust_score']:.3f}")
        print(f"  Source Reliability: {source_result['source_reliability_score']:.3f}")
        print(f"  Behavioral Risk: {source_result['behavioral_risk_flag']}")
    except Exception as e:
        print(f"✗ Phase 3 failed: {e}")
        return
    
    # Test Phase 4
    print("\nTesting Phase 4: Image Analysis...")
    try:
        image_analyzer = ImageAnalyzer()
        image_result = image_analyzer.process_post(
            post['post_id'],
            post.get('image_path')
        )
        print("✓ Phase 4 completed")
        print(f"  Image Tampered: {image_result['image_tampered']}")
        print(f"  AI Generated Prob: {image_result['ai_generated_probability']:.3f}")
        print(f"  OCR Text Length: {len(image_result['ocr_text'])}")
    except Exception as e:
        print(f"✗ Phase 4 failed: {e}")
        print("  (This is expected if Tesseract OCR is not installed)")
        return
    
    print("\n" + "="*60)
    print("✓ All phases tested successfully!")
    print("="*60)

if __name__ == "__main__":
    test_single_post()
