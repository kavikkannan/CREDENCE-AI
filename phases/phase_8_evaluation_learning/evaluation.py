"""
Phase 8: Evaluation and Learning Pipeline
Evaluates system performance and documents results.

This module implements:
- Performance metrics computation (accuracy, precision, recall)
- Error analysis
- Improvement opportunity logging

All outputs are stored in metrics_report.json.
Handles small evaluation datasets gracefully.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
import numpy as np


class PerformanceEvaluator:
    """
    Evaluates system performance using quantitative and qualitative metrics.
    """
    
    def __init__(self):
        """Initialize the performance evaluator."""
        pass
    
    def compute_metrics(self, 
                      predictions: List[float],
                      ground_truth: List[str],
                      threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute accuracy, precision, and recall metrics.
        
        Args:
            predictions: List of predicted credibility scores (0-1)
            ground_truth: List of ground truth labels ("true", "false", or "uncertain")
            threshold: Threshold for binary classification (default: 0.5)
            
        Returns:
            Dictionary with accuracy, precision, recall
        """
        if len(predictions) == 0 or len(ground_truth) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
        
        # Convert predictions to binary (credible if >= threshold)
        pred_binary = [1 if p >= threshold else 0 for p in predictions]
        
        # Convert ground truth to binary
        # "true" = 1 (credible), "false" = 0 (not credible), "uncertain" = exclude
        true_binary = []
        pred_filtered = []
        
        for i, gt in enumerate(ground_truth):
            if gt.lower() == "true":
                true_binary.append(1)
                pred_filtered.append(pred_binary[i])
            elif gt.lower() == "false":
                true_binary.append(0)
                pred_filtered.append(pred_binary[i])
            # Skip "uncertain" labels
        
        if len(true_binary) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
        
        # Calculate confusion matrix
        tp = sum(1 for p, t in zip(pred_filtered, true_binary) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(pred_filtered, true_binary) if p == 1 and t == 0)
        tn = sum(1 for p, t in zip(pred_filtered, true_binary) if p == 0 and t == 0)
        fn = sum(1 for p, t in zip(pred_filtered, true_binary) if p == 0 and t == 1)
        
        # Calculate metrics
        accuracy = (tp + tn) / len(true_binary) if len(true_binary) > 0 else 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        }
    
    def compute_detailed_metrics(self,
                                 predictions: List[float],
                                 ground_truth: List[str],
                                 threshold: float = 0.5) -> Dict[str, Any]:
        """
        Compute detailed metrics including confusion matrix.
        
        Args:
            predictions: List of predicted credibility scores (0-1)
            ground_truth: List of ground truth labels
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with detailed metrics
        """
        if len(predictions) == 0 or len(ground_truth) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                "total_samples": 0,
                "uncertain_samples": 0
            }
        
        pred_binary = [1 if p >= threshold else 0 for p in predictions]
        
        true_binary = []
        pred_filtered = []
        uncertain_count = 0
        
        for i, gt in enumerate(ground_truth):
            if gt.lower() == "true":
                true_binary.append(1)
                pred_filtered.append(pred_binary[i])
            elif gt.lower() == "false":
                true_binary.append(0)
                pred_filtered.append(pred_binary[i])
            else:
                uncertain_count += 1
        
        if len(true_binary) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                "total_samples": len(predictions),
                "uncertain_samples": uncertain_count
            }
        
        tp = sum(1 for p, t in zip(pred_filtered, true_binary) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(pred_filtered, true_binary) if p == 1 and t == 0)
        tn = sum(1 for p, t in zip(pred_filtered, true_binary) if p == 0 and t == 0)
        fn = sum(1 for p, t in zip(pred_filtered, true_binary) if p == 0 and t == 1)
        
        accuracy = (tp + tn) / len(true_binary) if len(true_binary) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            },
            "total_samples": len(predictions),
            "uncertain_samples": uncertain_count,
            "evaluated_samples": len(true_binary)
        }


class ErrorAnalyzer:
    """
    Performs qualitative error analysis to identify improvement opportunities.
    """
    
    def __init__(self):
        """Initialize the error analyzer."""
        pass
    
    def analyze_errors(self,
                      posts: List[Dict[str, Any]],
                      predictions: List[float],
                      ground_truth: List[str],
                      threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze errors and identify patterns.
        
        Args:
            posts: List of post dictionaries with all signals
            predictions: List of predicted credibility scores
            ground_truth: List of ground truth labels
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary with error analysis results
        """
        errors = {
            "false_positives": [],
            "false_negatives": [],
            "error_patterns": {},
            "improvement_opportunities": []
        }
        
        pred_binary = [1 if p >= threshold else 0 for p in predictions]
        
        for i, (post, pred, pred_bin, gt) in enumerate(zip(posts, predictions, pred_binary, ground_truth)):
            if gt.lower() == "uncertain":
                continue
            
            gt_binary = 1 if gt.lower() == "true" else 0
            
            # False positive: predicted credible but actually not credible
            if pred_bin == 1 and gt_binary == 0:
                errors["false_positives"].append({
                    "post_id": post.get("post_id", f"post_{i}"),
                    "predicted_score": round(pred, 4),
                    "ground_truth": gt,
                    "signals": self._extract_key_signals(post)
                })
            
            # False negative: predicted not credible but actually credible
            elif pred_bin == 0 and gt_binary == 1:
                errors["false_negatives"].append({
                    "post_id": post.get("post_id", f"post_{i}"),
                    "predicted_score": round(pred, 4),
                    "ground_truth": gt,
                    "signals": self._extract_key_signals(post)
                })
        
        # Identify error patterns
        errors["error_patterns"] = self._identify_patterns(errors)
        
        # Generate improvement opportunities
        errors["improvement_opportunities"] = self._generate_improvements(errors)
        
        return errors
    
    def _extract_key_signals(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key signals from a post for error analysis.
        
        Args:
            post: Post dictionary
            
        Returns:
            Dictionary with key signals
        """
        signals = {}
        
        # NLP signals
        nlp = post.get("nlp_signals", {})
        if nlp:
            signals["nlp"] = {
                "sentiment": nlp.get("sentiment"),
                "clickbait": nlp.get("clickbait"),
                "emotion": nlp.get("emotion")
            }
        
        # Source signals
        source = post.get("source_signals", {})
        if source:
            signals["source"] = {
                "account_trust": source.get("account_trust_score"),
                "source_reliability": source.get("source_reliability_score"),
                "behavioral_risk": source.get("behavioral_risk_flag")
            }
        
        # Image signals
        image = post.get("image_signals")
        if image:
            signals["image"] = {
                "tampered": image.get("image_tampered"),
                "ai_generated_prob": image.get("ai_generated_probability")
            }
        
        return signals
    
    def _identify_patterns(self, errors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify common patterns in errors.
        
        Args:
            errors: Error analysis dictionary
            
        Returns:
            Dictionary with identified patterns
        """
        patterns = {
            "false_positive_patterns": [],
            "false_negative_patterns": []
        }
        
        # Analyze false positives
        if errors["false_positives"]:
            fp_clickbait = sum(1 for e in errors["false_positives"] 
                             if e.get("signals", {}).get("nlp", {}).get("clickbait") == True)
            fp_low_trust = sum(1 for e in errors["false_positives"]
                             if e.get("signals", {}).get("source", {}).get("account_trust", 0.5) < 0.3)
            
            if fp_clickbait > len(errors["false_positives"]) * 0.5:
                patterns["false_positive_patterns"].append(
                    "Many false positives have clickbait patterns - may need stricter clickbait penalty"
                )
            if fp_low_trust > len(errors["false_positives"]) * 0.5:
                patterns["false_positive_patterns"].append(
                    "Many false positives have low account trust - may need stronger trust weighting"
                )
        
        # Analyze false negatives
        if errors["false_negatives"]:
            fn_high_trust = sum(1 for e in errors["false_negatives"]
                              if e.get("signals", {}).get("source", {}).get("account_trust", 0.5) > 0.7)
            fn_positive_sentiment = sum(1 for e in errors["false_negatives"]
                                       if e.get("signals", {}).get("nlp", {}).get("sentiment") == "POSITIVE")
            
            if fn_high_trust > len(errors["false_negatives"]) * 0.5:
                patterns["false_negative_patterns"].append(
                    "Many false negatives have high account trust - may be over-penalizing other factors"
                )
            if fn_positive_sentiment > len(errors["false_negatives"]) * 0.5:
                patterns["false_negative_patterns"].append(
                    "Many false negatives have positive sentiment - sentiment analysis may need adjustment"
                )
        
        return patterns
    
    def _generate_improvements(self, errors: Dict[str, Any]) -> List[str]:
        """
        Generate improvement opportunities based on error analysis.
        
        Args:
            errors: Error analysis dictionary
            
        Returns:
            List of improvement opportunity strings
        """
        improvements = []
        
        fp_count = len(errors["false_positives"])
        fn_count = len(errors["false_negatives"])
        
        if fp_count > fn_count * 2:
            improvements.append(
                "System tends to over-predict credibility. Consider increasing threshold or strengthening negative signal weights."
            )
        elif fn_count > fp_count * 2:
            improvements.append(
                "System tends to under-predict credibility. Consider decreasing threshold or strengthening positive signal weights."
            )
        
        # Add pattern-based improvements
        patterns = errors.get("error_patterns", {})
        improvements.extend(patterns.get("false_positive_patterns", []))
        improvements.extend(patterns.get("false_negative_patterns", []))
        
        if not improvements:
            improvements.append("Error patterns are balanced. Continue monitoring for edge cases.")
        
        return improvements


class EvaluationLearningPipeline:
    """
    Main pipeline for evaluation and learning.
    """
    
    def __init__(self):
        """Initialize the evaluation learning pipeline."""
        self.evaluator = PerformanceEvaluator()
        self.error_analyzer = ErrorAnalyzer()
    
    def process_evaluation(self,
                          posts: List[Dict[str, Any]],
                          output_file: str = "metrics_report.json") -> Dict[str, Any]:
        """
        Process evaluation data and generate metrics report.
        
        Args:
            posts: List of post dictionaries with final_decision and optional feedback_record
            output_file: Path to output JSON file
            
        Returns:
            Dictionary with evaluation metrics and analysis
        """
        # Extract predictions and ground truth
        predictions = []
        ground_truth = []
        valid_posts = []
        
        for post in posts:
            final_decision = post.get("final_decision", {})
            feedback_record = post.get("feedback_record") or post.get("user_feedback")
            
            if final_decision and feedback_record:
                pred_score = final_decision.get("final_credibility_score", 0.5)
                predictions.append(pred_score)
                
                # Get ground truth from feedback
                if isinstance(feedback_record, dict):
                    gt = feedback_record.get("user_feedback", "uncertain")
                else:
                    gt = str(feedback_record).lower()
                
                ground_truth.append(gt)
                valid_posts.append(post)
        
        if len(predictions) == 0:
            return {
                "evaluation_metrics": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0
                },
                "error_analysis": {
                    "false_positives": [],
                    "false_negatives": [],
                    "error_patterns": {},
                    "improvement_opportunities": []
                },
                "summary": "No evaluation data available",
                "timestamp": datetime.now().isoformat()
            }
        
        # Compute metrics
        metrics = self.evaluator.compute_detailed_metrics(predictions, ground_truth)
        
        # Perform error analysis
        error_analysis = self.error_analyzer.analyze_errors(
            valid_posts, predictions, ground_truth
        )
        
        # Generate report
        report = {
            "evaluation_metrics": {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"]
            },
            "detailed_metrics": {
                "confusion_matrix": metrics["confusion_matrix"],
                "total_samples": metrics["total_samples"],
                "uncertain_samples": metrics["uncertain_samples"],
                "evaluated_samples": metrics["evaluated_samples"]
            },
            "error_analysis": error_analysis,
            "summary": self._generate_summary(metrics, error_analysis),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save report
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nEvaluation report saved to {output_file}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Evaluated samples: {metrics['evaluated_samples']}")
        
        return report
    
    def _generate_summary(self, metrics: Dict[str, Any], error_analysis: Dict[str, Any]) -> str:
        """
        Generate qualitative summary of evaluation results.
        
        Args:
            metrics: Metrics dictionary
            error_analysis: Error analysis dictionary
            
        Returns:
            Summary string
        """
        accuracy = metrics["accuracy"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        
        summary_parts = []
        
        # Overall performance
        if accuracy >= 0.8:
            summary_parts.append("System shows strong overall performance.")
        elif accuracy >= 0.6:
            summary_parts.append("System shows moderate performance with room for improvement.")
        else:
            summary_parts.append("System performance needs significant improvement.")
        
        # Precision/Recall balance
        if precision > recall + 0.2:
            summary_parts.append("System is conservative (high precision, lower recall).")
        elif recall > precision + 0.2:
            summary_parts.append("System is permissive (high recall, lower precision).")
        else:
            summary_parts.append("System shows balanced precision and recall.")
        
        # Error patterns
        fp_count = len(error_analysis.get("false_positives", []))
        fn_count = len(error_analysis.get("false_negatives", []))
        
        if fp_count > fn_count:
            summary_parts.append(f"More false positives ({fp_count}) than false negatives ({fn_count}).")
        elif fn_count > fp_count:
            summary_parts.append(f"More false negatives ({fn_count}) than false positives ({fp_count}).")
        
        # Improvement opportunities
        improvements = error_analysis.get("improvement_opportunities", [])
        if improvements:
            summary_parts.append(f"Identified {len(improvements)} improvement opportunities.")
        
        return " ".join(summary_parts)


def main():
    """
    Main entry point for the evaluation learning pipeline.
    Can be used for testing or standalone execution.
    """
    import sys
    
    # Initialize pipeline
    pipeline = EvaluationLearningPipeline()
    
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
        
        output_file = sys.argv[2] if len(sys.argv) > 2 else "metrics_report.json"
        pipeline.process_evaluation(posts, output_file)
    else:
        # Run with sample data
        sample_posts = [
            {
                "post_id": "sample_1",
                "final_decision": {
                    "final_credibility_score": 0.825,
                    "agent_agreement_level": 0.85
                },
                "feedback_record": {
                    "user_feedback": "true"
                },
                "nlp_signals": {
                    "sentiment": "POSITIVE",
                    "clickbait": False,
                    "emotion": "joy"
                },
                "source_signals": {
                    "account_trust_score": 0.9,
                    "source_reliability_score": 0.8,
                    "behavioral_risk_flag": False
                }
            },
            {
                "post_id": "sample_2",
                "final_decision": {
                    "final_credibility_score": 0.225,
                    "agent_agreement_level": 0.70
                },
                "feedback_record": {
                    "user_feedback": "false"
                },
                "nlp_signals": {
                    "sentiment": "NEGATIVE",
                    "clickbait": True,
                    "emotion": "anger"
                },
                "source_signals": {
                    "account_trust_score": 0.1,
                    "source_reliability_score": 0.2,
                    "behavioral_risk_flag": True
                }
            }
        ]
        
        print("Running with sample data...")
        report = pipeline.process_evaluation(sample_posts, "metrics_report.json")
        print("\nEvaluation report:")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
