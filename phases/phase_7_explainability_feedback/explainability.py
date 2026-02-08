"""
Phase 7: Explainability and Feedback Pipeline
Generates human-readable explanations and captures user feedback.

This module implements:
- Human-readable explanation generation
- Warning label generation
- User feedback capture and logging

All outputs are stored keyed by post_id and match the user_facing_output schema.
Explanations are human-readable and do not expose raw model logits.
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime


class ExplainabilityGenerator:
    """
    Generates human-readable explanations from final decisions.
    Does not expose raw model logits or internal technical details.
    """
    
    def __init__(self):
        """Initialize the explainability generator."""
        pass
    
    def generate_explanation(self, final_decision: Dict[str, Any]) -> List[str]:
        """
        Generate human-readable explanation from final decision.
        
        Args:
            final_decision: Dictionary with final_credibility_score, agent_agreement_level, reasoning_trace
            
        Returns:
            List of human-readable explanation strings
        """
        explanations = []
        
        credibility_score = final_decision.get('final_credibility_score', 0.5)
        agent_agreement = final_decision.get('agent_agreement_level', 0.5)
        reasoning_trace = final_decision.get('reasoning_trace', [])
        
        # Main credibility assessment
        if credibility_score >= 0.7:
            explanations.append(f"This content appears to be credible (confidence: {credibility_score:.0%}).")
        elif credibility_score >= 0.4:
            explanations.append(f"This content has moderate credibility (confidence: {credibility_score:.0%}).")
        else:
            explanations.append(f"This content has low credibility (confidence: {credibility_score:.0%}).")
        
        # Agent agreement level
        if agent_agreement >= 0.8:
            explanations.append("Our analysis systems are in strong agreement about this assessment.")
        elif agent_agreement >= 0.5:
            explanations.append("Our analysis systems show moderate agreement about this assessment.")
        else:
            explanations.append("Our analysis systems show some disagreement, indicating this content may be ambiguous.")
        
        # Extract key insights from reasoning trace (human-readable)
        key_insights = self._extract_key_insights(reasoning_trace)
        explanations.extend(key_insights)
        
        return explanations
    
    def _extract_key_insights(self, reasoning_trace: List[str]) -> List[str]:
        """
        Extract key insights from reasoning trace and convert to human-readable format.
        
        Args:
            reasoning_trace: List of technical reasoning strings
            
        Returns:
            List of human-readable insight strings
        """
        insights = []
        
        # Analyze text signals
        text_insights = []
        for step in reasoning_trace:
            if "TextAgent" in step:
                if "Positive sentiment" in step:
                    text_insights.append("The text has a positive tone.")
                elif "Negative sentiment" in step:
                    text_insights.append("The text has a negative tone.")
                if "Clickbait pattern" in step:
                    text_insights.append("The text contains clickbait patterns.")
                if "Manipulative emotion" in step:
                    emotion = step.split(":")[-1].strip() if ":" in step else ""
                    text_insights.append(f"The text uses manipulative emotional language ({emotion}).")
                elif "Positive emotion" in step:
                    emotion = step.split(":")[-1].strip() if ":" in step else ""
                    text_insights.append(f"The text conveys positive emotions ({emotion}).")
        
        if text_insights:
            insights.append("Text Analysis: " + " ".join(text_insights[:2]))  # Limit to 2 key points
        
        # Analyze image signals
        image_insights = []
        for step in reasoning_trace:
            if "ImageAgent" in step:
                if "Image tampering detected" in step:
                    image_insights.append("The image shows signs of manipulation.")
                elif "No image tampering" in step:
                    image_insights.append("The image appears authentic.")
                if "High AI-generated probability" in step:
                    image_insights.append("The image may be AI-generated.")
                elif "Low AI-generated probability" in step:
                    image_insights.append("The image appears to be naturally created.")
        
        if image_insights:
            insights.append("Image Analysis: " + " ".join(image_insights[:2]))  # Limit to 2 key points
        
        # Analyze source signals
        source_insights = []
        for step in reasoning_trace:
            if "SourceAgent" in step:
                if "Account trust score" in step:
                    # Extract score from step
                    try:
                        score_part = step.split(":")[1].split("(")[0].strip()
                        score = float(score_part)
                        if score >= 0.7:
                            source_insights.append("The account appears trustworthy.")
                        elif score <= 0.3:
                            source_insights.append("The account has low trust indicators.")
                    except:
                        pass
                if "Behavioral risk flag" in step:
                    source_insights.append("The account shows suspicious behavioral patterns.")
                elif "No behavioral risk flags" in step:
                    source_insights.append("The account shows normal behavioral patterns.")
        
        if source_insights:
            insights.append("Source Analysis: " + " ".join(source_insights[:2]))  # Limit to 2 key points
        
        return insights
    
    def generate_warning_label(self, credibility_score: float) -> str:
        """
        Generate warning label based on credibility score.
        
        Args:
            credibility_score: Final credibility score (0-1)
            
        Returns:
            Warning label string
        """
        if credibility_score >= 0.7:
            return "Credible"
        elif credibility_score >= 0.5:
            return "Caution Advised"
        elif credibility_score >= 0.3:
            return "Low Credibility"
        else:
            return "High Risk - Verify Information"
    
    def format_credibility_score(self, score: float) -> float:
        """
        Format credibility score for user-facing output.
        Rounds to 2 decimal places for readability.
        
        Args:
            score: Raw credibility score
            
        Returns:
            Formatted score
        """
        return round(score, 2)


class FeedbackLogger:
    """
    Logs and stores user feedback for evaluation and learning.
    """
    
    def __init__(self, feedback_file: str = "feedback_log.json"):
        """
        Initialize the feedback logger.
        
        Args:
            feedback_file: Path to feedback log file
        """
        self.feedback_file = feedback_file
    
    def log_feedback(self, 
                    post_id: str,
                    user_feedback: str,
                    final_decision: Dict[str, Any],
                    user_facing_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log user feedback for a post.
        
        Args:
            post_id: Unique identifier for the post
            user_feedback: User feedback ("true", "false", or "uncertain")
            final_decision: Final decision from phase 6
            user_facing_output: User-facing output from this phase
            
        Returns:
            Feedback record dictionary
        """
        feedback_record = {
            "post_id": post_id,
            "timestamp": datetime.now().isoformat(),
            "user_feedback": user_feedback,
            "system_prediction": {
                "credibility_score": user_facing_output.get("credibility_score"),
                "warning_label": user_facing_output.get("warning_label")
            },
            "final_decision": {
                "final_credibility_score": final_decision.get("final_credibility_score"),
                "agent_agreement_level": final_decision.get("agent_agreement_level")
            }
        }
        
        # Save to file
        self._save_feedback(feedback_record)
        
        return feedback_record
    
    def _save_feedback(self, feedback_record: Dict[str, Any]):
        """
        Save feedback record to file.
        
        Args:
            feedback_record: Feedback record to save
        """
        # Load existing feedback if file exists
        feedback_log = []
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    feedback_log = json.load(f)
            except:
                feedback_log = []
        
        # Append new feedback
        feedback_log.append(feedback_record)
        
        # Save back to file
        output_dir = os.path.dirname(self.feedback_file) if os.path.dirname(self.feedback_file) else '.'
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_log, f, indent=2, ensure_ascii=False)


class ExplainabilityFeedbackPipeline:
    """
    Main pipeline for explainability and feedback.
    """
    
    def __init__(self, feedback_file: str = "feedback_log.json"):
        """
        Initialize the explainability feedback pipeline.
        
        Args:
            feedback_file: Path to feedback log file
        """
        self.explainer = ExplainabilityGenerator()
        self.feedback_logger = FeedbackLogger(feedback_file)
    
    def process_post(self,
                    post_id: str,
                    final_decision: Dict[str, Any],
                    user_feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single post and generate user-facing output.
        
        Args:
            post_id: Unique identifier for the post
            final_decision: Final decision from phase 6
            user_feedback: Optional user feedback ("true", "false", or "uncertain")
            
        Returns:
            Dictionary with user_facing_output and optional feedback_record
        """
        credibility_score = final_decision.get('final_credibility_score', 0.5)
        
        # Generate explanation
        explanation = self.explainer.generate_explanation(final_decision)
        
        # Generate warning label
        warning_label = self.explainer.generate_warning_label(credibility_score)
        
        # Format credibility score for user
        formatted_score = self.explainer.format_credibility_score(credibility_score)
        
        # Create user-facing output
        user_facing_output = {
            "credibility_score": formatted_score,
            "warning_label": warning_label,
            "explanation": explanation
        }
        
        result = {
            "user_facing_output": user_facing_output
        }
        
        # Log feedback if provided
        if user_feedback:
            feedback_record = self.feedback_logger.log_feedback(
                post_id, user_feedback, final_decision, user_facing_output
            )
            result["feedback_record"] = feedback_record
        
        return result
    
    def process_batch(self, posts: List[Dict[str, Any]], 
                     output_file: str = "final_output.json"):
        """
        Process multiple posts and save results keyed by post_id.
        
        Args:
            posts: List of post dictionaries with final_decision and optional user_feedback
            output_file: Path to output JSON file
        """
        results = {}
        
        for i, post in enumerate(posts, 1):
            post_id = post.get('post_id', f'post_{i}')
            final_decision = post.get('final_decision', {})
            user_feedback = post.get('user_feedback')  # Optional
            
            print(f"Processing post {i}/{len(posts)}: {post_id}")
            
            try:
                result = self.process_post(post_id, final_decision, user_feedback)
                results[post_id] = result
            except Exception as e:
                print(f"Error processing post {post_id}: {e}")
                # Return default output on error
                results[post_id] = {
                    "user_facing_output": {
                        "credibility_score": 0.5,
                        "warning_label": "Caution Advised",
                        "explanation": ["Unable to generate explanation due to processing error."]
                    }
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
    Main entry point for the explainability feedback pipeline.
    Can be used for testing or standalone execution.
    """
    import sys
    
    # Initialize pipeline
    feedback_file = sys.argv[3] if len(sys.argv) > 3 else "feedback_log.json"
    pipeline = ExplainabilityFeedbackPipeline(feedback_file=feedback_file)
    
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
        
        output_file = sys.argv[2] if len(sys.argv) > 2 else "final_output.json"
        pipeline.process_batch(posts, output_file)
    else:
        # Run with sample data
        sample_posts = [
            {
                "post_id": "sample_1",
                "final_decision": {
                    "final_credibility_score": 0.825,
                    "agent_agreement_level": 0.85,
                    "reasoning_trace": [
                        "TextAgent: Positive sentiment detected (+0.1)",
                        "TextAgent: No clickbait patterns detected (+0.05)",
                        "TextAgent: Positive emotion detected: joy (+0.05)",
                        "ImageAgent: No image tampering detected (+0.1)",
                        "SourceAgent: Account trust score: 0.900 (impact: 0.160)",
                        "SourceAgent: No behavioral risk flags (+0.05)"
                    ]
                },
                "user_feedback": "true"  # Optional
            },
            {
                "post_id": "sample_2",
                "final_decision": {
                    "final_credibility_score": 0.225,
                    "agent_agreement_level": 0.70,
                    "reasoning_trace": [
                        "TextAgent: Negative sentiment detected (-0.1)",
                        "TextAgent: Clickbait pattern detected (-0.2)",
                        "TextAgent: Manipulative emotion detected: anger (-0.15)",
                        "ImageAgent: Image tampering detected (-0.3)",
                        "SourceAgent: Behavioral risk flag detected (-0.2)"
                    ]
                },
                "user_feedback": "false"  # Optional
            }
        ]
        
        print("Running with sample data...")
        results = pipeline.process_batch(sample_posts, "final_output.json")
        print("\nSample results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
