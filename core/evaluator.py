"""
RAGAS-based evaluation system for measuring search quality.
Evaluates retrieval and answer generation performance using standard metrics.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from dataclasses import dataclass
from core.search_router import SearchRouter
from utils.utils import get_logger
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy, context_precision, faithfulness, answer_correctness
)

logger = get_logger(__name__)

@dataclass
class RAGEvaluationMetrics:
    """Container for RAGAS evaluation metrics with scoring"""
    answer_relevancy: float     # How relevant the answer is to the question
    context_precision: float    # Precision of retrieved context
    faithfulness: float         # Factual consistency with context
    answer_correctness: float   # Overall answer quality
    overall_score: float        # Weighted average of all metrics
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format"""
        return {
            'answer_relevancy': self.answer_relevancy,
            'context_precision': self.context_precision,
            'faithfulness': self.faithfulness,
            'answer_correctness': self.answer_correctness,
            'overall_score': self.overall_score
        }

class RAGEvaluator:
    """RAGAS-powered search quality evaluator with history tracking"""
    
    def __init__(self):
        """Initialize RAGAS metrics and evaluation tracking"""
        self.ragas_metrics = [
            answer_relevancy, context_precision, faithfulness, answer_correctness
        ]
        self.evaluation_history = []  # Track all evaluations
    
    def evaluate_search_quality(self, query: str, expected_skills: List[str], 
                              search_mode: str = "deep", top_k: int = 5) -> RAGEvaluationMetrics:
        """Run search and evaluate results using RAGAS quality metrics"""
        
        try:
            # Create search router instance
            search_router = SearchRouter()
            search_result = search_router.search(query, top_k, search_mode=search_mode)
            
            if not search_result.get('results'):
                return self.create_default_metrics()
            
            candidates = search_result['results']
            return self.evaluate_with_ragas(query, candidates, expected_skills)
                
        except Exception as e:
            logger.error(f"Search quality evaluation failed: {e}")
            return self.create_default_metrics()
    
    def create_default_metrics(self) -> RAGEvaluationMetrics:
        """Return zero metrics when search fails or returns no results"""
        return RAGEvaluationMetrics(
            answer_relevancy=0.0,
            context_precision=0.0,
            faithfulness=0.0,
            answer_correctness=0.0,
            overall_score=0.0
        )
    
    def evaluate_with_ragas(self, query: str, candidates: List[Dict], 
                            expected_skills: List[str]) -> RAGEvaluationMetrics:
        """Run RAGAS evaluation on search results and calculate final metrics"""
        
        evaluation_data = self.prepare_ragas_data(query, candidates, expected_skills)
        results = evaluate(evaluation_data, metrics=self.ragas_metrics)
        
        metrics = RAGEvaluationMetrics(
            answer_relevancy=float(results['answer_relevancy']),
            context_precision=float(results['context_precision']),
            faithfulness=float(results['faithfulness']),
            answer_correctness=float(results['answer_correctness']),
            overall_score=0.0
        )
        
        metrics.overall_score = self.calculate_overall_score(metrics)
        self.store_evaluation(query, metrics)
        
        return metrics
    
    def prepare_ragas_data(self, query: str, candidates: List[Dict], 
                           expected_skills: List[str]) -> pd.DataFrame:
        """Convert search results to RAGAS-compatible DataFrame format"""
        
        data = []
        for candidate in candidates:
            metadata = candidate.get('metadata', {})
            
            # Create context from candidate metadata
            context_parts = []
            if 'name' in metadata:
                context_parts.append(f"Name: {metadata['name']}")
            if 'skills' in metadata:
                context_parts.append(f"Skills: {', '.join(metadata['skills'][:5])}")
            if 'experience' in metadata:
                context_parts.append(f"Experience: {metadata['experience']} years")
            
            context = " | ".join(context_parts) if context_parts else "No metadata"
            
            # Create ground truth and generated answer
            ground_truth = self.create_ground_truth(metadata.get('skills', []), expected_skills)
            generated_answer = f"{metadata.get('name', 'Candidate')} with skills in {', '.join(metadata.get('skills', [])[:3])}"
            
            data.append({
                'question': query,
                'contexts': [context],
                'ground_truth': ground_truth,
                'answer': generated_answer
            })
        
        return pd.DataFrame(data)
    
    def create_ground_truth(self, candidate_skills: List[str], expected_skills: List[str]) -> str:
        """Generate ground truth labels based on skill matching percentage"""
        if not expected_skills:
            return "No skills specified"
         
        matched_skills = [skill for skill in expected_skills 
                         if skill.lower() in [s.lower() for s in candidate_skills]]
        match_percentage = len(matched_skills) / len(expected_skills)
        
        if match_percentage >= 0.8:
            return f"Excellent match: {len(matched_skills)}/{len(expected_skills)} skills"
        elif match_percentage >= 0.6:
            return f"Good match: {len(matched_skills)}/{len(expected_skills)} skills"
        elif match_percentage >= 0.4:
            return f"Moderate match: {len(matched_skills)}/{len(expected_skills)} skills"
        else:
            return f"Poor match: {len(matched_skills)}/{len(expected_skills)} skills"
    
    def calculate_overall_score(self, metrics: RAGEvaluationMetrics) -> float:
        """Compute weighted average of all RAGAS metrics"""
        weights = {
            'answer_relevancy': 0.30,
            'context_precision': 0.30,
            'faithfulness': 0.20,
            'answer_correctness': 0.20
        }
        
        overall_score = (
            metrics.answer_relevancy * weights['answer_relevancy'] +
            metrics.context_precision * weights['context_precision'] +
            metrics.faithfulness * weights['faithfulness'] +
            metrics.answer_correctness * weights['answer_correctness']
        )
        
        return overall_score
    
    def store_evaluation(self, query: str, metrics: RAGEvaluationMetrics):
        """Save evaluation results to history for later analysis"""
        evaluation_record = {
            'query': query,
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics.to_dict()
        }
        
        self.evaluation_history.append(evaluation_record)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations performed"""
        if not self.evaluation_history:
            return {"message": "No evaluations performed yet"}
        
        # Calculate averages
        avg_metrics = {}
        for metric in ['answer_relevancy', 'context_precision', 'faithfulness', 
                      'answer_correctness', 'overall_score']:
            values = [record['metrics'][metric] for record in self.evaluation_history]
            avg_metrics[f'avg_{metric}'] = sum(values) / len(values)
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'average_metrics': avg_metrics,
            'recent_evaluations': self.evaluation_history[-5:]
        }
    
    def export_evaluations(self, filename: str = "rag_evaluations.csv") -> bool:
        """Export all evaluation history to CSV file for analysis"""
        if not self.evaluation_history:
            return False
        
        try:
            export_data = []
            for i, record in enumerate(self.evaluation_history):
                row = {
                    'evaluation_id': i + 1,
                    'query': record['query'],
                    'timestamp': record['timestamp'],
                    **record['metrics']
                }
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            return True
            
        except Exception as e:
            logger.error(f"Failed to export evaluations: {e}")
            return False
