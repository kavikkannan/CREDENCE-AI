# Phase 4: Image Analysis Pipeline

## Overview
This phase analyzes visual content for manipulation and hidden textual cues using computer vision techniques.

## Inputs
- `image_path`: String or null (path to image file)

### Optional Inputs
- `image_metadata`: Object (optional metadata about the image)

## Outputs (image_signals schema)
- `ocr_text`: String (text extracted from image using OCR)
- `image_tampered`: Boolean (indicates if image tampering detected)
- `ai_generated_probability`: Float (0-1) (probability that image is AI-generated)

## Processing Methods

### OCR Text Extraction
- Uses Tesseract OCR (pytesseract)
- Extracts text from images
- Returns empty string if no text found or image_path is null

### Image Tampering Detection (ELA)
- Uses Error Level Analysis (ELA) technique
- Compares compression levels to detect modifications
- Heuristics:
  - High mean ELA values (>15)
  - High standard deviation (>10)
  - High max values (>50)
- Returns True if tampering detected, False otherwise

### AI-Generated Image Probability
- Rule-based heuristics (simulated for prototype)
- Factors considered:
  1. **Smoothness**: Very low Laplacian variance indicates unusual smoothness
  2. **Symmetry**: High symmetry scores (some AI generators create symmetric images)
  3. **Color Distribution**: Low entropy in color histograms (uniform distributions)
- Returns probability between 0 and 1

## Usage

### Prerequisites
1. Install Tesseract OCR:
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

2. Install Python dependencies:
```bash
pip install opencv-python pytesseract pillow scikit-image numpy
```

### Standalone execution
```bash
python image_analysis.py input.json image_outputs.json
```

### Programmatic usage
```python
from image_analysis import ImageAnalyzer

# Initialize analyzer
# On Windows, specify tesseract path if needed:
# analyzer = ImageAnalyzer(tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
analyzer = ImageAnalyzer()

# Process single post
result = analyzer.process_post(
    post_id="post_123",
    image_path="path/to/image.jpg"
)

# Process batch
posts = [
    {
        "post_id": "1",
        "image_path": "path/to/image1.jpg"
    },
    {
        "post_id": "2",
        "image_path": None  # Will return default values
    }
]
analyzer.process_batch(posts, "image_outputs.json")
```

## Handling Null Image Paths
- If `image_path` is null, empty, or "null", the phase returns:
  - `ocr_text`: ""
  - `image_tampered`: False
  - `ai_generated_probability`: 0.0
- The phase gracefully skips processing when no image is available

## Constraints
- Uses opencv, pytesseract, and scikit-image
- Skips phase if image_path is null
- AI-generated probability is simulated/rule-based (suitable for prototype)
- All outputs stored keyed by `post_id`
- Outputs match `image_signals` schema exactly
- Follows phase contracts strictly

## Notes
- ELA tampering detection works best with JPEG images
- OCR accuracy depends on image quality and text clarity
- AI-generated probability estimation uses heuristics and may not be as accurate as specialized models
