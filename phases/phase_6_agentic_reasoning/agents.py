"""
Phase 6: Agentic Reasoning Pipeline
Performs multi-agent reasoning and conflict resolution across all signals.

This module implements:
- TextAgent: Evaluates text-based credibility signals
- ImageAgent: Evaluates image-based credibility signals
- SourceAgent: Evaluates source-based credibility signals
- AggregatorAgent: Combines agent outputs with weighted aggregation

All outputs are stored keyed by post_id and match the final_decision schema.
"""

import json
import os
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent reasoning system.
    """
    
    def __init__(self, name: str):
        """
        Initialize the agent.
        
        Args:
            name: Name of the agent
        """
        self.name = name
        self.reasoning_trace = []
    
    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Evaluate credibility based on agent-specific signals.
        
        Returns:
            Dictionary with credibility_score and reasoning
        """
        pass
    
    def add_reasoning(self, message: str):
        """
        Add a reasoning step to the trace.
        
        Args:
            message: Reasoning message to add
        """
        self.reasoning_trace.append(f"{self.name}: {message}")


class TextAgent(BaseAgent):
    """
    Agent that evaluates credibility based on text/NLP signals.
    """
    
    def __init__(self):
        super().__init__("TextAgent")
    
    def evaluate(self, nlp_signals: Dict[str, Any], 
                misinformation_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate credibility based on NLP signals.
        
        Args:
            nlp_signals: NLP signals from phase 2
            misinformation_assessment: Misinformation assessment from phase 5
            
        Returns:
            Dictionary with credibility_score and reasoning
        """
        self.reasoning_trace = []
        score = 0.5  # Start with neutral
        
        # Factor 1: Sentiment analysis
        sentiment = nlp_signals.get('sentiment', '').upper()
        if sentiment == 'POSITIVE':
            score += 0.1
            self.add_reasoning("Positive sentiment detected (+0.1)")
        elif sentiment == 'NEGATIVE':
            score -= 0.1
            self.add_reasoning("Negative sentiment detected (-0.1)")
        else:
            self.add_reasoning("Neutral sentiment")
        
        # Factor 2: Clickbait detection
        if nlp_signals.get('clickbait', False):
            score -= 0.2
            self.add_reasoning("Clickbait pattern detected (-0.2)")
        else:
            score += 0.05
            self.add_reasoning("No clickbait patterns detected (+0.05)")
        
        # Factor 3: Emotion analysis
        emotion = nlp_signals.get('emotion', '').lower()
        manipulative_emotions = ['anger', 'fear', 'disgust']
        if emotion in manipulative_emotions:
            score -= 0.15
            self.add_reasoning(f"Manipulative emotion detected: {emotion} (-0.15)")
        elif emotion in ['joy', 'surprise']:
            score += 0.05
            self.add_reasoning(f"Positive emotion detected: {emotion} (+0.05)")
        else:
            self.add_reasoning(f"Emotion: {emotion}")
        
        # Factor 4: Misinformation assessment from phase 5
        credibility_from_phase5 = misinformation_assessment.get('content_credibility_score', 0.5)
        score = (score + credibility_from_phase5) / 2.0  # Average with phase 5 assessment
        self.add_reasoning(f"Integrated phase 5 credibility score: {credibility_from_phase5:.3f}")
        
        # Normalize to 0-1
        score = max(0.0, min(1.0, score))
        
        return {
            "credibility_score": round(score, 4),
            "reasoning": self.reasoning_trace.copy()
        }


class ImageAgent(BaseAgent):
    """
    Agent that evaluates credibility based on image signals.
    """
    
    def __init__(self):
        super().__init__("ImageAgent")
    
    def evaluate(self, image_signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate credibility based on image signals.
        
        Args:
            image_signals: Image signals from phase 4 (optional)
            
        Returns:
            Dictionary with credibility_score and reasoning
        """
        self.reasoning_trace = []
        
        # If no image signals, return neutral score
        if not image_signals:
            self.add_reasoning("No image signals available, defaulting to neutral")
            return {
                "credibility_score": 0.5,
                "reasoning": self.reasoning_trace.copy()
            }
        
        score = 0.5  # Start with neutral
        
        # Factor 1: Image tampering
        if image_signals.get('image_tampered', False):
            score -= 0.3
            self.add_reasoning("Image tampering detected (-0.3)")
        else:
            score += 0.1
            self.add_reasoning("No image tampering detected (+0.1)")
        
        # Factor 2: AI-generated probability
        ai_prob = image_signals.get('ai_generated_probability', 0.0)
        if ai_prob > 0.7:
            score -= 0.25
            self.add_reasoning(f"High AI-generated probability: {ai_prob:.3f} (-0.25)")
        elif ai_prob > 0.4:
            score -= 0.1
            self.add_reasoning(f"Moderate AI-generated probability: {ai_prob:.3f} (-0.1)")
        elif ai_prob < 0.2:
            score += 0.05
            self.add_reasoning(f"Low AI-generated probability: {ai_prob:.3f} (+0.05)")
        else:
            self.add_reasoning(f"AI-generated probability: {ai_prob:.3f}")
        
        # Factor 3: OCR text (if available, could indicate manipulated content)
        ocr_text = image_signals.get('ocr_text', '')
        if ocr_text and len(ocr_text) > 10:
            # OCR text exists - could be legitimate or suspicious
            # For now, neutral impact
            self.add_reasoning(f"OCR text extracted ({len(ocr_text)} chars)")
        
        # Normalize to 0-1
        score = max(0.0, min(1.0, score))
        
        return {
            "credibility_score": round(score, 4),
            "reasoning": self.reasoning_trace.copy()
        }


class SourceAgent(BaseAgent):
    """
    Agent that evaluates credibility based on source/account signals.
    """
    
    def __init__(self):
        super().__init__("SourceAgent")
    
    def evaluate(self, source_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate credibility based on source signals.
        
        Args:
            source_signals: Source signals from phase 3
            
        Returns:
            Dictionary with credibility_score and reasoning
        """
        self.reasoning_trace = []
        score = 0.5  # Start with neutral
        
        # Factor 1: Account trust score
        account_trust = source_signals.get('account_trust_score', 0.5)
        score += (account_trust - 0.5) * 0.4  # Scale impact
        self.add_reasoning(f"Account trust score: {account_trust:.3f} (impact: {(account_trust - 0.5) * 0.4:.3f})")
        
        # Factor 2: Source reliability score
        source_reliability = source_signals.get('source_reliability_score', 0.5)
        score += (source_reliability - 0.5) * 0.4  # Scale impact
        self.add_reasoning(f"Source reliability score: {source_reliability:.3f} (impact: {(source_reliability - 0.5) * 0.4:.3f})")
        
        # Factor 3: Behavioral risk flag
        if source_signals.get('behavioral_risk_flag', False):
            score -= 0.2
            self.add_reasoning("Behavioral risk flag detected (-0.2)")
        else:
            score += 0.05
            self.add_reasoning("No behavioral risk flags (+0.05)")
        
        # Normalize to 0-1
        score = max(0.0, min(1.0, score))
        
        return {
            "credibility_score": round(score, 4),
            "reasoning": self.reasoning_trace.copy()
        }


class AggregatorAgent(BaseAgent):
    """
    Agent that aggregates outputs from all other agents.
    """
    
    def __init__(self, agent_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the aggregator agent.
        
        Args:
            agent_weights: Optional dictionary of agent weights (default: equal weights)
        """
        super().__init__("AggregatorAgent")
        self.agent_weights = agent_weights or {
            "TextAgent": 0.35,
            "ImageAgent": 0.25,
            "SourceAgent": 0.40
        }
        # Normalize weights
        total_weight = sum(self.agent_weights.values())
        self.agent_weights = {k: v / total_weight for k, v in self.agent_weights.items()}
    
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """
        Evaluate method required by BaseAgent.
        For AggregatorAgent, this calls aggregate().
        
        Args:
            **kwargs: Should contain 'agent_outputs' key
            
        Returns:
            Dictionary with aggregated results
        """
        agent_outputs = kwargs.get('agent_outputs', {})
        return self.aggregate(agent_outputs)
    
    def aggregate(self, agent_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate agent outputs with weighted combination.
        
        Args:
            agent_outputs: Dictionary mapping agent names to their outputs
            
        Returns:
            Dictionary with final_credibility_score, agent_agreement_level, and reasoning
        """
        self.reasoning_trace = []
        
        # Collect scores and reasoning from all agents
        scores = {}
        all_reasoning = []
        
        for agent_name, output in agent_outputs.items():
            if agent_name in self.agent_weights:
                scores[agent_name] = output.get('credibility_score', 0.5)
                all_reasoning.extend(output.get('reasoning', []))
                self.add_reasoning(f"{agent_name} score: {scores[agent_name]:.3f} (weight: {self.agent_weights[agent_name]:.3f})")
        
        # Calculate weighted average
        if not scores:
            self.add_reasoning("No agent scores available, defaulting to neutral")
            return {
                "final_credibility_score": 0.5,
                "agent_agreement_level": 0.0,
                "reasoning": self.reasoning_trace.copy()
            }
        
        weighted_sum = sum(scores[agent] * self.agent_weights[agent] 
                          for agent in scores if agent in self.agent_weights)
        final_score = weighted_sum
        
        self.add_reasoning(f"Weighted aggregation: {final_score:.3f}")
        
        # Calculate agent agreement level (1 - standard deviation of scores)
        if len(scores) > 1:
            score_values = list(scores.values())
            std_dev = np.std(score_values)
            # Normalize std_dev (max possible is 0.5 for scores in [0,1])
            agreement_level = max(0.0, 1.0 - (std_dev / 0.5))
        else:
            agreement_level = 1.0  # Perfect agreement if only one agent
        
        self.add_reasoning(f"Agent agreement level: {agreement_level:.3f} (std_dev: {np.std(list(scores.values())):.3f})")
        
        # Combine all reasoning traces
        combined_reasoning = all_reasoning + self.reasoning_trace
        
        return {
            "final_credibility_score": round(final_score, 4),
            "agent_agreement_level": round(agreement_level, 4),
            "reasoning": combined_reasoning
        }


class MultiAgentReasoningSystem:
    """
    Main system that coordinates all agents for multi-agent reasoning.
    """
    
    def __init__(self, agent_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the multi-agent reasoning system.
        
        Args:
            agent_weights: Optional dictionary of agent weights
        """
        self.text_agent = TextAgent()
        self.image_agent = ImageAgent()
        self.source_agent = SourceAgent()
        self.aggregator_agent = AggregatorAgent(agent_weights)
    
    def process_post(self,
                    post_id: str,
                    nlp_signals: Dict[str, Any],
                    source_signals: Dict[str, Any],
                    image_signals: Optional[Dict[str, Any]] = None,
                    misinformation_assessment: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a single post through all agents.
        
        Args:
            post_id: Unique identifier for the post
            nlp_signals: NLP signals from phase 2
            source_signals: Source signals from phase 3
            image_signals: Image signals from phase 4 (optional)
            misinformation_assessment: Misinformation assessment from phase 5
            
        Returns:
            Dictionary matching final_decision schema
        """
        if misinformation_assessment is None:
            misinformation_assessment = {"content_credibility_score": 0.5, "risk_category": "medium"}
        
        # Run each agent independently
        text_output = self.text_agent.evaluate(nlp_signals, misinformation_assessment)
        image_output = self.image_agent.evaluate(image_signals)
        source_output = self.source_agent.evaluate(source_signals)
        
        # Aggregate agent outputs
        agent_outputs = {
            "TextAgent": text_output,
            "ImageAgent": image_output,
            "SourceAgent": source_output
        }
        
        final_decision = self.aggregator_agent.aggregate(agent_outputs)
        
        return final_decision
    
    def process_batch(self, posts: List[Dict[str, Any]], 
                     output_file: str = "agent_outputs.json"):
        """
        Process multiple posts and save results keyed by post_id.
        
        Args:
            posts: List of post dictionaries with all required signals
            output_file: Path to output JSON file
        """
        results = {}
        
        for i, post in enumerate(posts, 1):
            post_id = post.get('post_id', f'post_{i}')
            
            # Extract signals from previous phases
            nlp_signals = post.get('nlp_signals', {})
            source_signals = post.get('source_signals', {})
            image_signals = post.get('image_signals')  # Optional
            misinformation_assessment = post.get('misinformation_assessment', {})
            
            print(f"Processing post {i}/{len(posts)}: {post_id}")
            
            try:
                final_decision = self.process_post(
                    post_id, nlp_signals, source_signals, 
                    image_signals, misinformation_assessment
                )
                results[post_id] = final_decision
            except Exception as e:
                print(f"Error processing post {post_id}: {e}")
                # Return default decision on error
                results[post_id] = {
                    "final_credibility_score": 0.5,
                    "agent_agreement_level": 0.0,
                    "reasoning": [f"Error: {str(e)}"]
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
    Main entry point for the agentic reasoning pipeline.
    Can be used for testing or standalone execution.
    """
    import sys
    
    # Initialize system with optional custom weights
    agent_weights = {
        "TextAgent": 0.35,
        "ImageAgent": 0.25,
        "SourceAgent": 0.40
    }
    
    system = MultiAgentReasoningSystem(agent_weights=agent_weights)
    
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
        
        output_file = sys.argv[2] if len(sys.argv) > 2 else "agent_outputs.json"
        system.process_batch(posts, output_file)
    else:
        # Run with sample data
        sample_posts = [
            {
                "post_id": "sample_1",
                "nlp_signals": {
                    "sentiment": "POSITIVE",
                    "emotion": "joy",
                    "clickbait": False,
                    "extracted_claim": "Example claim",
                    "text_embedding_id": "abc123"
                },
                "source_signals": {
                    "account_trust_score": 0.9,
                    "source_reliability_score": 0.8,
                    "behavioral_risk_flag": False
                },
                "image_signals": {
                    "ocr_text": "",
                    "image_tampered": False,
                    "ai_generated_probability": 0.1
                },
                "misinformation_assessment": {
                    "content_credibility_score": 0.75,
                    "risk_category": "low"
                }
            },
            {
                "post_id": "sample_2",
                "nlp_signals": {
                    "sentiment": "NEGATIVE",
                    "emotion": "anger",
                    "clickbait": True,
                    "extracted_claim": "Suspicious claim",
                    "text_embedding_id": "def456"
                },
                "source_signals": {
                    "account_trust_score": 0.1,
                    "source_reliability_score": 0.2,
                    "behavioral_risk_flag": True
                },
                "image_signals": {
                    "ocr_text": "FAKE",
                    "image_tampered": True,
                    "ai_generated_probability": 0.8
                },
                "misinformation_assessment": {
                    "content_credibility_score": 0.25,
                    "risk_category": "high"
                }
            }
        ]
        
        print("Running with sample data...")
        results = system.process_batch(sample_posts, "agent_outputs.json")
        print("\nSample results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
