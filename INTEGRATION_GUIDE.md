# Frontend-Backend Integration Guide

This guide explains how to run the complete GenAI Social Media Credibility Analyzer with the Next.js frontend connected to the Python FastAPI backend.

## Architecture

- **Backend**: Python FastAPI server (`api_server.py`) running on `http://localhost:8000`
- **Frontend**: Next.js application running on `http://localhost:3000`
- **Pipeline**: All 8 phases of the credibility analysis pipeline

## Setup Instructions

### 1. Backend Setup (Python)

1. **Activate virtual environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install API dependencies** (if not already installed):
   ```powershell
   pip install fastapi uvicorn[standard] python-multipart
   ```

3. **Start the API server**:
   ```powershell
   python api_server.py
   ```
   
   The server will start on `http://localhost:8000`
   
   **Note**: On first run, models will be downloaded (this may take several minutes).

### 2. Frontend Setup (Next.js)

1. **Navigate to frontend directory**:
   ```powershell
   cd frontend
   ```

2. **Install dependencies** (if not already installed):
   ```powershell
   npm install
   ```

3. **Start the development server**:
   ```powershell
   npm run dev
   ```
   
   The frontend will start on `http://localhost:3000`

## Usage

1. **Open the frontend**: Navigate to `http://localhost:3000` in your browser

2. **Load Dataset**: The frontend will automatically load posts from `data/twitter_dataset.json` via the API

3. **Select a Post**: Click on any post in the "Input Data Explorer" view

4. **Run Analysis**: Click "Run Simulation" to analyze the selected post through all 8 phases

5. **View Results**: After analysis completes, view the detailed results including:
   - Credibility score
   - Warning labels
   - Agentic reasoning trace
   - NLP signals
   - Source trust scores

## API Endpoints

The backend provides the following endpoints:

- `GET /` - API information
- `GET /health` - Health check
- `GET /api/dataset` - Get all posts from dataset
- `POST /api/analyze-single` - Analyze a single post
- `POST /api/analyze` - Analyze multiple posts
- `GET /api/results/{post_id}` - Get saved results for a post

## Troubleshooting

### API Not Connected
- Make sure the Python backend is running on port 8000
- Check that `api_server.py` started without errors
- Verify models loaded successfully (check console output)

### CORS Errors
- The backend is configured to allow requests from `localhost:3000`
- If using a different port, update `allow_origins` in `api_server.py`

### Image Path Issues
- Images should be in `data/media/image/` directory
- Image paths in the dataset are automatically mapped to this location

### Model Loading Issues
- First run will download models (can take 5-10 minutes)
- Ensure sufficient disk space (models require ~2GB)
- Check internet connection for model downloads

## Development

### Backend Development
- API server auto-reloads on code changes (if using `uvicorn --reload`)
- Check `http://localhost:8000/docs` for interactive API documentation

### Frontend Development
- Next.js hot-reloads on code changes
- Check browser console for errors
- API calls are logged in the browser console

## Production Deployment

For production:
1. Build the frontend: `npm run build` in the frontend directory
2. Use a production WSGI server for the backend (e.g., Gunicorn)
3. Configure proper CORS settings
4. Set up environment variables for API URLs

## File Structure

```
creditability-analyzer/
├── api_server.py              # FastAPI backend server
├── run_pipeline.py            # Standalone pipeline runner
├── phases/                    # All pipeline phases
│   ├── phase_2_nlp_analysis/
│   ├── phase_3_source_account_analysis/
│   ├── phase_4_image_analysis/
│   ├── phase_5_misinformation_modeling/
│   ├── phase_6_agentic_reasoning/
│   ├── phase_7_explainability_feedback/
│   └── phase_8_evaluation_learning/
├── frontend/                  # Next.js frontend
│   └── app/
│       ├── page.tsx          # Main dashboard
│       └── api.ts            # API service
├── data/                      # Input data
│   ├── twitter_dataset.json
│   └── media/image/          # Image files
└── outputs/                   # Pipeline outputs
