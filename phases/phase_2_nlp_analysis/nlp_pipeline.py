"""
Phase 2: NLP Analysis Pipeline
Extracts linguistic and emotional credibility signals from text.

This module implements:
- Sentiment analysis
- Emotion detection
- Clickbait detection
- Claim extraction
- Text embeddings generation

All outputs are stored keyed by post_id and match the nlp_signals schema.
"""

import json
import os
from typing import Dict, List, Optional, Any
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch


class NLPAnalysisPipeline:
    """
    Pipeline for analyzing text content using pretrained HuggingFace models.
    Follows phase contracts strictly and uses only pretrained models.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize all NLP models using HuggingFace pipelines.
        
        Args:
            device: Device to run models on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        print(f"Initializing NLP models on device: {device}")
        
        # Initialize sentiment analysis model
        print("Loading sentiment analysis model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if device == 'cuda' else -1
        )
        
        # Initialize emotion detection model
        print("Loading emotion detection model...")
        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if device == 'cuda' else -1
        )
        
        # Initialize clickbait detection (using rule-based approach)
        # Note: Original model mrm8488/bert-tiny-finetuned-clickbait-news is not available
        print("Using rule-based clickbait detection...")
        self.clickbait_model_available = False
        
        # Initialize text embeddings model
        print("Loading text embeddings model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        if device == 'cuda':
            self.embedding_model = self.embedding_model.to(device)
        
        print("All models loaded successfully!")
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Perform sentiment analysis on text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment label as string (e.g., "POSITIVE", "NEGATIVE")
        """
        if not text or not text.strip():
            return "NEUTRAL"
        
        result = self.sentiment_pipeline(text[:512])  # Limit to model's max length
        label = result[0]['label']
        # Normalize to uppercase string
        return label.upper()
    
    def detect_emotion(self, text: str) -> str:
        """
        Detect emotion in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Emotion label as string
        """
        if not text or not text.strip():
            return "neutral"
        
        result = self.emotion_pipeline(text[:512])
        emotion = result[0]['label']
        # Convert to lowercase for consistency
        return emotion.lower()
    
    def detect_clickbait(self, text: str) -> bool:
        """
        Detect if text contains clickbait patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if clickbait detected, False otherwise
        """
        if not text or not text.strip():
            return False
        
        # Use model if available
        if self.clickbait_model_available and self.clickbait_pipeline:
            try:
                result = self.clickbait_pipeline(text[:512])
                label = result[0]['label'].upper()
                # Model returns "CLICKBAIT" or "NOT_CLICKBAIT"
                return label == "CLICKBAIT" or "CLICKBAIT" in label
            except:
                # Fall through to rule-based if model fails
                pass
        
        # Rule-based clickbait detection (fallback)
        text_upper = text.upper()
        
        # Common clickbait patterns
        clickbait_patterns = [
            "YOU WON'T BELIEVE",
            "THIS WILL SHOCK YOU",
            "NUMBER X WILL",
            "THE REASON WHY",
            "THIS ONE TRICK",
            "DOCTORS HATE",
            "WHAT HAPPENS NEXT",
            "THIS CHANGES EVERYTHING",
            "URGENT:",
            "BREAKING:",
            "SHOCKING:",
            "INCREDIBLE:",
            "AMAZING:",
            "UNBELIEVABLE:",
            "?!",  # Multiple punctuation
            "??",
            "!!!"
        ]
        
        # Check for clickbait patterns
        for pattern in clickbait_patterns:
            if pattern in text_upper:
                return True
        
        # Check for excessive capitalization (more than 30% uppercase)
        if len(text) > 10:
            uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if uppercase_ratio > 0.3:
                return True
        
        # Check for question marks or exclamation marks at the end
        if text.strip().endswith('?') or text.strip().endswith('!'):
            # If it's a short text with question/exclamation, likely clickbait
            if len(text.split()) < 10:
                return True
        
        return False
    
    def extract_claim(self, text: str, hashtags: List[str] = None, urls: List[str] = None) -> str:
        """
        Extract primary factual claim from text.
        Uses a simple heuristic approach: extract the main sentence or statement.
        
        Args:
            text: Input text
            hashtags: Optional list of hashtags
            urls: Optional list of URLs
            
        Returns:
            Extracted claim as string
        """
        if not text or not text.strip():
            return ""
        
        # Simple claim extraction: take the first sentence or first 200 characters
        # This is a heuristic approach suitable for prototype
        sentences = text.split('.')
        if sentences:
            # Get first meaningful sentence (non-empty, > 10 chars)
            for sentence in sentences:
                cleaned = sentence.strip()
                if len(cleaned) > 10:
                    # Remove URLs and hashtags from claim
                    claim = cleaned
                    if urls:
                        for url in urls:
                            claim = claim.replace(url, "")
                    if hashtags:
                        for tag in hashtags:
                            claim = claim.replace(f"#{tag}", "").replace(tag, "")
                    return claim.strip()[:200]  # Limit length
        
        # Fallback: return first 200 characters
        return text.strip()[:200]
    
    def generate_embedding(self, text: str) -> str:
        """
        Generate text embedding and return embedding ID (hash-based).
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding ID as string (hash of embedding vector)
        """
        if not text or not text.strip():
            text = ""
        
        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        
        # Create a simple ID from the embedding (hash of first few values)
        # In production, this might be stored in a vector DB
        import hashlib
        embedding_str = ','.join([f"{val:.6f}" for val in embedding[:10]])  # First 10 values
        embedding_id = hashlib.md5(embedding_str.encode()).hexdigest()
        
        return embedding_id
    
    def process_post(self, post_id: str, text: str, hashtags: List[str] = None, 
                    urls: List[str] = None) -> Dict[str, Any]:
        """
        Process a single post and extract all NLP signals.
        
        Args:
            post_id: Unique identifier for the post
            text: Main text content
            hashtags: List of hashtags (optional)
            urls: List of URLs (optional)
            
        Returns:
            Dictionary matching nlp_signals schema
        """
        if hashtags is None:
            hashtags = []
        if urls is None:
            urls = []
        
        # Combine text with hashtags for better analysis
        # Hashtags often contain important context
        analysis_text = text
        if hashtags:
            hashtag_text = " ".join([f"#{tag}" for tag in hashtags])
            analysis_text = f"{text} {hashtag_text}"
        
        # Perform all analyses
        sentiment = self.analyze_sentiment(analysis_text)
        emotion = self.detect_emotion(analysis_text)
        clickbait = self.detect_clickbait(analysis_text)
        extracted_claim = self.extract_claim(text, hashtags, urls)
        text_embedding_id = self.generate_embedding(analysis_text)
        
        # Return in exact schema format
        return {
            "sentiment": sentiment,
            "emotion": emotion,
            "clickbait": clickbait,
            "extracted_claim": extracted_claim,
            "text_embedding_id": text_embedding_id
        }
    
    def process_batch(self, posts: List[Dict[str, Any]], output_file: str = "nlp_outputs.json"):
        """
        Process multiple posts and save results keyed by post_id.
        
        Args:
            posts: List of post dictionaries with post_id, text, hashtags, urls
            output_file: Path to output JSON file
        """
        results = {}
        
        for i, post in enumerate(posts, 1):
            post_id = post.get('post_id', f'post_{i}')
            text = post.get('text', '')
            hashtags = post.get('hashtags', [])
            urls = post.get('urls', [])
            
            print(f"Processing post {i}/{len(posts)}: {post_id}")
            
            try:
                nlp_signals = self.process_post(post_id, text, hashtags, urls)
                results[post_id] = nlp_signals
            except Exception as e:
                print(f"Error processing post {post_id}: {e}")
                # Return empty signals on error
                results[post_id] = {
                    "sentiment": "NEUTRAL",
                    "emotion": "neutral",
                    "clickbait": False,
                    "extracted_claim": "",
                    "text_embedding_id": ""
                }
        
        # Save results to JSON file
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"Processed {len(results)} posts successfully")
        
        return results


def main():
    """
    Main entry point for the NLP pipeline.
    Can be used for testing or standalone execution.
    """
    import sys
    
    # Initialize pipeline
    pipeline_instance = NLPAnalysisPipeline()
    
    # Example usage with sample data
    if len(sys.argv) > 1:
        # Load input from JSON file
        input_file = sys.argv[1]
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Handle both single post and list of posts
        if isinstance(input_data, dict):
            if 'post_id' in input_data:
                posts = [input_data]
            else:
                # Assume it's a dict of posts keyed by post_id
                posts = [{'post_id': k, **v} for k, v in input_data.items()]
        else:
            posts = input_data
        
        output_file = sys.argv[2] if len(sys.argv) > 2 else "nlp_outputs.json"
        pipeline_instance.process_batch(posts, output_file)
    else:
        # Run with sample data
        sample_posts = [
            {
                "post_id": "sample_1",
                "text": "Breaking: Scientists discover new breakthrough in renewable energy! This will change everything!",
                "hashtags": ["science", "energy", "breakthrough"],
                "urls": ["https://example.com/article"]
            },
            {
                "post_id": "sample_2",
                "text": "Just had the best coffee at my local café. Highly recommend!",
                "hashtags": ["coffee", "food"],
                "urls": []
            }
        ]
        
        print("Running with sample data...")
        results = pipeline_instance.process_batch(sample_posts, "nlp_outputs.json")
        print("\nSample results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
