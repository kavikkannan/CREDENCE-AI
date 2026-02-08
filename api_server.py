"""
FastAPI Backend Server for GenAI Social Media Credibility Analyzer
Provides REST API endpoints to interact with the pipeline.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
import sys
from pathlib import Path

# Add phases to path
sys.path.insert(0, str(Path(__file__).parent))

from phases.phase_2_nlp_analysis.nlp_pipeline import NLPAnalysisPipeline
from phases.phase_3_source_account_analysis.source_scoring import SourceAccountAnalyzer
from phases.phase_4_image_analysis.image_analysis import ImageAnalyzer
from phases.phase_5_misinformation_modeling.misinformation_model import MisinformationModel
from phases.phase_6_agentic_reasoning.agents import MultiAgentReasoningSystem
from phases.phase_7_explainability_feedback.explainability import ExplainabilityFeedbackPipeline

app = FastAPI(title="GenAI Social Media Credibility Analyzer API")

# CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instances (initialized on startup)
nlp_pipeline = None
source_analyzer = None
image_analyzer = None
misinformation_model = None
agentic_system = None
explainability_pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline components on startup"""
    global nlp_pipeline, source_analyzer, image_analyzer, misinformation_model, agentic_system, explainability_pipeline
    
    print("Initializing pipeline components...")
    try:
        nlp_pipeline = NLPAnalysisPipeline()
        source_analyzer = SourceAccountAnalyzer()
        image_analyzer = ImageAnalyzer()
        misinformation_model = MisinformationModel()
        agentic_system = MultiAgentReasoningSystem()
        explainability_pipeline = ExplainabilityFeedbackPipeline()
        print("All pipeline components initialized successfully!")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        raise


# Pydantic models for request/response
class AccountInput(BaseModel):
    account_id: str
    account_age_days: int
    verified: bool
    historical_post_count: int
    name: Optional[str] = None
    screen_name: Optional[str] = None
    description: Optional[str] = None
    followers_count: Optional[int] = None
    friends_count: Optional[int] = None
    profile_image_url: Optional[str] = None

class PostInput(BaseModel):
    post_id: str
    platform: str
    text: str
    image_path: Optional[str] = None
    urls: List[str] = []
    hashtags: List[str] = []
    likes: int = 0
    retweets: int = 0
    timestamp: str
    account: AccountInput


class AnalyzeRequest(BaseModel):
    posts: List[PostInput]


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GenAI Social Media Credibility Analyzer API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "components_loaded": all([
        nlp_pipeline is not None,
        source_analyzer is not None,
        image_analyzer is not None,
        misinformation_model is not None,
        agentic_system is not None,
        explainability_pipeline is not None
    ])}


@app.post("/api/analyze")
async def analyze_posts(request: AnalyzeRequest):
    """
    Analyze one or more posts through the complete pipeline.
    """
    try:
        results = {}
        
        for post in request.posts:
            post_id = post.post_id
            
            # Map image path if provided
            image_path = None
            if post.image_path:
                # Check if it's already a full path
                if os.path.isabs(post.image_path) or post.image_path.startswith('data/'):
                    image_path = post.image_path
                else:
                    # Map to data/media/image/ directory
                    image_path = os.path.join('data', 'media', 'image', post.image_path)
                    if not os.path.exists(image_path):
                        image_path = None
            
            # Phase 2: NLP Analysis
            nlp_result = nlp_pipeline.process_post(
                post_id, post.text, post.hashtags, post.urls
            )
            
            # Phase 3: Source Account Analysis
            account = post.account if isinstance(post.account, dict) else post.account.dict()
            source_result = source_analyzer.process_post(
                post_id,
                account.get('account_age_days', 0),
                account.get('verified', False),
                account.get('historical_post_count', 0),
                post.urls,
                account.get('name'),
                account.get('screen_name'),
                account.get('description'),
                account.get('followers_count')
            )
            
            # Phase 4: Image Analysis
            image_result = image_analyzer.process_post(post_id, image_path)
            
            # Phase 5: Misinformation Modeling
            misinformation_result = misinformation_model.process_post(
                post_id,
                nlp_result,
                source_result,
                image_result,
                post.text
            )
            
            # Phase 6: Agentic Reasoning
            agentic_result = agentic_system.process_post(
                post_id,
                nlp_result,
                source_result,
                image_result,
                misinformation_result
            )
            
            # Phase 7: Explainability
            explainability_result = explainability_pipeline.process_post(
                post_id,
                agentic_result
            )
            
            # Combine all results
            results[post_id] = {
                "nlp_signals": nlp_result,
                "source_signals": source_result,
                "image_signals": image_result,
                "misinformation_assessment": misinformation_result,
                "final_decision": agentic_result,
                "user_facing_output": explainability_result.get("user_facing_output", {})
            }
        
        return {"results": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze-single")
async def analyze_single_post(post: PostInput):
    """
    Analyze a single post through the complete pipeline.
    """
    request = AnalyzeRequest(posts=[post])
    result = await analyze_posts(request)
    return result["results"][post.post_id]


@app.get("/api/dataset")
async def get_dataset():
    """
    Get the sample dataset from twitter_dataset.json
    """
    try:
        dataset_path = "data/twitter_dataset.json"
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {"dataset": data, "count": len(data)}
        else:
            return {"dataset": [], "count": 0, "error": "Dataset file not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")


@app.get("/api/results/{post_id}")
async def get_results(post_id: str):
    """
    Get analysis results for a specific post from saved outputs.
    """
    try:
        # Try to load from phase 7 outputs
        results_file = "outputs/phase_7_explainability_outputs.json"
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                if post_id in all_results:
                    return all_results[post_id]
        
        raise HTTPException(status_code=404, detail=f"Results not found for {post_id}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading results: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
