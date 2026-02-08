"""
Phase 4: Image Analysis Pipeline
Analyzes visual content for manipulation and hidden textual cues.

This module implements:
- OCR text extraction
- Image tampering detection using ELA (Error Level Analysis)
- AI-generated image probability estimation

All outputs are stored keyed by post_id and match the image_signals schema.
Skips processing if image_path is null.
"""

import json
import os
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
from PIL import Image
import pytesseract
from skimage import filters, exposure
from skimage.metrics import structural_similarity as ssim


class ImageAnalyzer:
    """
    Analyzer for image content using computer vision techniques.
    Follows phase contracts strictly and uses rule-based/image processing methods.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the image analyzer.
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional, for Windows)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif os.name == 'nt':  # Windows
            # Try common Windows installation paths
            common_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files\Ttttt\tesseract.exe',  # Your current installation
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
    
    def extract_ocr_text(self, image_path: str) -> str:
        """
        Extract text from image using OCR (Tesseract).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        if not image_path or not os.path.exists(image_path):
            return ""
        
        try:
            # Read image using PIL
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang='eng')
            
            # Clean up text
            text = text.strip()
            
            return text
        except Exception as e:
            print(f"Error in OCR extraction: {e}")
            return ""
    
    def detect_image_tampering_ela(self, image_path: str, quality: int = 90) -> bool:
        """
        Detect image tampering using Error Level Analysis (ELA).
        ELA reveals areas of an image that have been modified by comparing
        compression levels.
        
        Args:
            image_path: Path to the image file
            quality: JPEG quality for ELA analysis (default: 90)
            
        Returns:
            True if tampering detected, False otherwise
        """
        if not image_path or not os.path.exists(image_path):
            return False
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Save image at specified quality
            temp_path = image_path + '_temp_ela.jpg'
            Image.fromarray(image_rgb).save(temp_path, 'JPEG', quality=quality)
            
            # Load the re-saved image
            resaved = cv2.imread(temp_path)
            resaved_rgb = cv2.cvtColor(resaved, cv2.COLOR_BGR2RGB)
            
            # Calculate difference (ELA)
            ela_image = np.abs(image_rgb.astype(np.float32) - resaved_rgb.astype(np.float32))
            ela_image = ela_image.astype(np.uint8)
            
            # Convert to grayscale for analysis
            ela_gray = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY)
            
            # Calculate statistics
            mean_ela = np.mean(ela_gray)
            std_ela = np.std(ela_gray)
            max_ela = np.max(ela_gray)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Heuristic: High variance or high max values indicate tampering
            # Thresholds are tuned based on typical ELA values
            tampering_threshold_mean = 15.0
            tampering_threshold_std = 10.0
            tampering_threshold_max = 50.0
            
            is_tampered = (
                mean_ela > tampering_threshold_mean or
                std_ela > tampering_threshold_std or
                max_ela > tampering_threshold_max
            )
            
            return bool(is_tampered)
            
        except Exception as e:
            print(f"Error in ELA tampering detection: {e}")
            return False
    
    def estimate_ai_generated_probability(self, image_path: str) -> float:
        """
        Estimate probability that image is AI-generated.
        Uses heuristics based on image characteristics.
        Note: This is a simulated/rule-based approach for prototype.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Probability (float between 0 and 1)
        """
        if not image_path or not os.path.exists(image_path):
            return 0.0
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Heuristic 1: Check for unusual smoothness (AI images often very smooth)
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Very low variance indicates unusual smoothness (potential AI)
            smoothness_score = 0.0
            if laplacian_var < 100:
                smoothness_score = 0.4
            elif laplacian_var < 200:
                smoothness_score = 0.2
            
            # Heuristic 2: Check for perfect symmetry (some AI generators create symmetric images)
            # Split image in half and compare
            h, w = gray.shape
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            
            # Resize if needed for comparison
            if right_half.shape != left_half.shape:
                right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
            
            if left_half.shape == right_half.shape:
                symmetry = ssim(left_half, right_half)
                symmetry_score = 0.0
                if symmetry > 0.9:
                    symmetry_score = 0.3
                elif symmetry > 0.85:
                    symmetry_score = 0.15
            else:
                symmetry_score = 0.0
            
            # Heuristic 3: Check for unusual color distribution
            # AI images sometimes have very uniform color distributions
            hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
            
            # Calculate entropy of histograms (uniform = lower entropy)
            def entropy(hist):
                hist = hist.flatten()
                hist = hist[hist > 0]  # Remove zeros
                if len(hist) == 0:
                    return 0
                prob = hist / hist.sum()
                return -np.sum(prob * np.log2(prob))
            
            entropy_r = entropy(hist_r)
            entropy_g = entropy(hist_g)
            entropy_b = entropy(hist_b)
            avg_entropy = (entropy_r + entropy_g + entropy_b) / 3.0
            
            color_score = 0.0
            if avg_entropy < 5.0:  # Very low entropy
                color_score = 0.3
            elif avg_entropy < 6.0:
                color_score = 0.15
            
            # Combine scores (normalized to 0-1)
            total_score = min(1.0, smoothness_score + symmetry_score + color_score)
            
            return round(total_score, 4)
            
        except Exception as e:
            print(f"Error in AI-generated probability estimation: {e}")
            return 0.0
    
    def process_post(self, post_id: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single post's image and extract all image signals.
        
        Args:
            post_id: Unique identifier for the post
            image_path: Path to the image file (can be None)
            
        Returns:
            Dictionary matching image_signals schema
        """
        # Skip if image_path is null
        if not image_path or image_path == "null" or image_path == "":
            return {
                "ocr_text": "",
                "image_tampered": False,
                "ai_generated_probability": 0.0
            }
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return {
                "ocr_text": "",
                "image_tampered": False,
                "ai_generated_probability": 0.0
            }
        
        # Perform all analyses
        ocr_text = self.extract_ocr_text(image_path)
        image_tampered = self.detect_image_tampering_ela(image_path)
        ai_generated_probability = self.estimate_ai_generated_probability(image_path)
        
        # Return in exact schema format
        return {
            "ocr_text": ocr_text,
            "image_tampered": image_tampered,
            "ai_generated_probability": ai_generated_probability
        }
    
    def process_batch(self, posts: List[Dict[str, Any]], 
                     output_file: str = "image_outputs.json"):
        """
        Process multiple posts and save results keyed by post_id.
        
        Args:
            posts: List of post dictionaries with image_path
            output_file: Path to output JSON file
        """
        results = {}
        
        for i, post in enumerate(posts, 1):
            post_id = post.get('post_id', f'post_{i}')
            image_path = post.get('image_path')
            
            print(f"Processing post {i}/{len(posts)}: {post_id}")
            
            try:
                image_signals = self.process_post(post_id, image_path)
                results[post_id] = image_signals
            except Exception as e:
                print(f"Error processing post {post_id}: {e}")
                # Return default signals on error
                results[post_id] = {
                    "ocr_text": "",
                    "image_tampered": False,
                    "ai_generated_probability": 0.0
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
    Main entry point for the image analysis pipeline.
    Can be used for testing or standalone execution.
    """
    import sys
    
    # Initialize analyzer
    # On Windows, you may need to specify tesseract path:
    # analyzer = ImageAnalyzer(tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    analyzer = ImageAnalyzer()
    
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
        
        output_file = sys.argv[2] if len(sys.argv) > 2 else "image_outputs.json"
        analyzer.process_batch(posts, output_file)
    else:
        # Run with sample data (using placeholder paths)
        sample_posts = [
            {
                "post_id": "sample_1",
                "image_path": "path/to/image1.jpg"
            },
            {
                "post_id": "sample_2",
                "image_path": None
            },
            {
                "post_id": "sample_3",
                "image_path": "path/to/image3.jpg"
            }
        ]
        
        print("Running with sample data...")
        print("Note: Sample paths are placeholders. Provide actual image paths for real analysis.")
        results = analyzer.process_batch(sample_posts, "image_outputs.json")
        print("\nSample results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
