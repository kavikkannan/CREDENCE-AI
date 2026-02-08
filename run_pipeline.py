"""
Master Pipeline Runner
Runs all phases sequentially with the Twitter dataset.
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
from phases.phase_5_misinformation_modeling.misinformation_model import MisinformationModel
from phases.phase_6_agentic_reasoning.agents import MultiAgentReasoningSystem
from phases.phase_7_explainability_feedback.explainability import ExplainabilityFeedbackPipeline
from phases.phase_8_evaluation_learning.evaluation import EvaluationLearningPipeline


def load_and_prepare_data(input_file: str) -> list:
    """
    Load dataset and prepare it for processing.
    Maps image paths correctly.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare data: map image paths correctly
    prepared_data = []
    for post in data:
        post_copy = post.copy()
        
        # Map image path to full path if image_path exists
        if post_copy.get('image_path'):
            image_filename = post_copy['image_path']
            # Check if it's already a full path
            if not os.path.isabs(image_filename) and not image_filename.startswith('data/'):
                # Map to data/media/image/ directory
                image_path = os.path.join('data', 'media', 'image', image_filename)
                post_copy['image_path'] = image_path
            # If file doesn't exist, set to None
            if not os.path.exists(post_copy['image_path']):
                print(f"Warning: Image not found for {post_copy['post_id']}: {post_copy['image_path']}")
                post_copy['image_path'] = None
        
        prepared_data.append(post_copy)
    
    return prepared_data


def run_phase_2_nlp(posts: list, output_dir: str = "outputs") -> dict:
    """Run Phase 2: NLP Analysis"""
    print("\n" + "="*60)
    print("PHASE 2: NLP Analysis")
    print("="*60)
    
    pipeline = NLPAnalysisPipeline()
    
    # Prepare input for phase 2
    phase2_input = []
    for post in posts:
        phase2_input.append({
            'post_id': post['post_id'],
            'text': post.get('text', ''),
            'hashtags': post.get('hashtags', []),
            'urls': post.get('urls', [])
        })
    
    output_file = os.path.join(output_dir, 'phase_2_nlp_outputs.json')
    results = pipeline.process_batch(phase2_input, output_file)
    
    return results


def run_phase_3_source(posts: list, output_dir: str = "outputs") -> dict:
    """Run Phase 3: Source Account Analysis"""
    print("\n" + "="*60)
    print("PHASE 3: Source Account Analysis")
    print("="*60)
    
    analyzer = SourceAccountAnalyzer()
    
    # Prepare input for phase 3
    phase3_input = []
    for post in posts:
        account = post.get('account', {})
        phase3_input.append({
            'post_id': post['post_id'],
            'account': account,  # Pass full account object
            'urls': post.get('urls', [])
        })
    
    output_file = os.path.join(output_dir, 'phase_3_source_outputs.json')
    results = analyzer.process_batch(phase3_input, output_file)
    
    return results


def run_phase_4_image(posts: list, output_dir: str = "outputs") -> dict:
    """Run Phase 4: Image Analysis"""
    print("\n" + "="*60)
    print("PHASE 4: Image Analysis")
    print("="*60)
    
    analyzer = ImageAnalyzer()
    
    # Prepare input for phase 4
    phase4_input = []
    for post in posts:
        phase4_input.append({
            'post_id': post['post_id'],
            'image_path': post.get('image_path')
        })
    
    output_file = os.path.join(output_dir, 'phase_4_image_outputs.json')
    results = analyzer.process_batch(phase4_input, output_file)
    
    return results


def run_phase_5_misinformation(posts: list, nlp_results: dict, source_results: dict, 
                               image_results: dict, output_dir: str = "outputs") -> dict:
    """Run Phase 5: Misinformation Modeling"""
    print("\n" + "="*60)
    print("PHASE 5: Misinformation Modeling")
    print("="*60)
    
    model = MisinformationModel()
    
    # Prepare input for phase 5
    phase5_input = []
    for post in posts:
        post_id = post['post_id']
        phase5_input.append({
            'post_id': post_id,
            'text': post.get('text', ''),
            'nlp_signals': nlp_results.get(post_id, {}),
            'source_signals': source_results.get(post_id, {}),
            'image_signals': image_results.get(post_id)  # Optional
        })
    
    output_file = os.path.join(output_dir, 'phase_5_misinformation_outputs.json')
    results = model.process_batch(phase5_input, output_file)
    
    return results


def run_phase_6_agentic(posts: list, nlp_results: dict, source_results: dict,
                        image_results: dict, misinformation_results: dict,
                        output_dir: str = "outputs") -> dict:
    """Run Phase 6: Agentic Reasoning"""
    print("\n" + "="*60)
    print("PHASE 6: Agentic Reasoning")
    print("="*60)
    
    system = MultiAgentReasoningSystem()
    
    # Prepare input for phase 6
    phase6_input = []
    for post in posts:
        post_id = post['post_id']
        phase6_input.append({
            'post_id': post_id,
            'nlp_signals': nlp_results.get(post_id, {}),
            'source_signals': source_results.get(post_id, {}),
            'image_signals': image_results.get(post_id),  # Optional
            'misinformation_assessment': misinformation_results.get(post_id, {})
        })
    
    output_file = os.path.join(output_dir, 'phase_6_agentic_outputs.json')
    results = system.process_batch(phase6_input, output_file)
    
    return results


def run_phase_7_explainability(agentic_results: dict, output_dir: str = "outputs") -> dict:
    """Run Phase 7: Explainability and Feedback"""
    print("\n" + "="*60)
    print("PHASE 7: Explainability and Feedback")
    print("="*60)
    
    pipeline = ExplainabilityFeedbackPipeline()
    
    # Prepare input for phase 7
    phase7_input = []
    for post_id, final_decision in agentic_results.items():
        phase7_input.append({
            'post_id': post_id,
            'final_decision': final_decision
        })
    
    output_file = os.path.join(output_dir, 'phase_7_explainability_outputs.json')
    results = pipeline.process_batch(phase7_input, output_file)
    
    return results


def run_phase_8_evaluation(posts: list, agentic_results: dict, explainability_results: dict,
                          output_dir: str = "outputs") -> dict:
    """Run Phase 8: Evaluation and Learning"""
    print("\n" + "="*60)
    print("PHASE 8: Evaluation and Learning")
    print("="*60)
    
    pipeline = EvaluationLearningPipeline()
    
    # Prepare input for phase 8 (combine all data)
    phase8_input = []
    for post in posts:
        post_id = post['post_id']
        phase8_input.append({
            'post_id': post_id,
            'final_decision': agentic_results.get(post_id, {}),
            'feedback_record': explainability_results.get(post_id, {}).get('feedback_record'),
            # Include signals for error analysis
            'nlp_signals': {},  # Could load from phase 2 if needed
            'source_signals': {},  # Could load from phase 3 if needed
            'image_signals': {}  # Could load from phase 4 if needed
        })
    
    output_file = os.path.join(output_dir, 'phase_8_evaluation_report.json')
    results = pipeline.process_evaluation(phase8_input, output_file)
    
    return results


def main():
    """Main pipeline execution"""
    # Configuration
    input_file = "data/twitter_dataset.json"
    output_dir = "outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("GENAI SOCIAL MEDIA CREDIBILITY ANALYZER")
    print("Master Pipeline Execution")
    print("="*60)
    
    # Load and prepare data
    print(f"\nLoading dataset from: {input_file}")
    posts = load_and_prepare_data(input_file)
    print(f"Loaded {len(posts)} posts")
    
    # Run all phases sequentially
    try:
        # Phase 2: NLP Analysis
        nlp_results = run_phase_2_nlp(posts, output_dir)
        
        # Phase 3: Source Account Analysis
        source_results = run_phase_3_source(posts, output_dir)
        
        # Phase 4: Image Analysis
        image_results = run_phase_4_image(posts, output_dir)
        
        # Phase 5: Misinformation Modeling
        misinformation_results = run_phase_5_misinformation(
            posts, nlp_results, source_results, image_results, output_dir
        )
        
        # Phase 6: Agentic Reasoning
        agentic_results = run_phase_6_agentic(
            posts, nlp_results, source_results, image_results, 
            misinformation_results, output_dir
        )
        
        # Phase 7: Explainability and Feedback
        explainability_results = run_phase_7_explainability(agentic_results, output_dir)
        
        # Phase 8: Evaluation and Learning
        evaluation_results = run_phase_8_evaluation(
            posts, agentic_results, explainability_results, output_dir
        )
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*60)
        print(f"\nAll outputs saved to: {output_dir}/")
        print("\nOutput files:")
        print(f"  - Phase 2: phase_2_nlp_outputs.json")
        print(f"  - Phase 3: phase_3_source_outputs.json")
        print(f"  - Phase 4: phase_4_image_outputs.json")
        print(f"  - Phase 5: phase_5_misinformation_outputs.json")
        print(f"  - Phase 6: phase_6_agentic_outputs.json")
        print(f"  - Phase 7: phase_7_explainability_outputs.json")
        print(f"  - Phase 8: phase_8_evaluation_report.json")
        
    except Exception as e:
        print(f"\nERROR: Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
