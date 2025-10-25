from typing import List, Dict, Tuple
import numpy as np

class JobRanker:
    """Handles ranking and filtering of job recommendations."""
    
    @staticmethod
    def filter_by_top_domains(jobs: List[Dict], top_domains: List[tuple], min_domain_score: float = 0.1) -> List[Dict]:
        """Filter jobs to keep only those from relevant domains."""
        if not jobs:
            return []
        
        relevant_domains = [domain.lower() for domain, prob in top_domains if prob >= min_domain_score]
        
        filtered_jobs = []
        for job in jobs:
            industry = job.get("industry", "").lower()
            if industry in relevant_domains:
                filtered_jobs.append(job)
        
        return filtered_jobs
    
    @staticmethod
    def domain_aware_reranking(
        jobs: List[Dict], 
        top_domains: List[tuple],
        user_embedding: np.ndarray,
        domain_weight: float = 0.6,
        diversity_penalty: float = 0.1
    ) -> List[Dict]:
        """Re-rank jobs using domain classification and similarity metrics."""
        if not jobs:
            return []
        
        domain_probs = {domain.lower(): prob for domain, prob in top_domains}
        
        for job in jobs:
            industry = job.get("industry", "").lower()
            
            domain_score = domain_probs.get(industry, 0.0)
            vector_score = float(job.get("vector_score", 0.0)) if job.get("vector_score") is not None else 0.0
            normalized_vector_score = min(max(vector_score, 0.0), 1.0)
            
            domain_bonus = domain_score * 0.3 if domain_score > 0.5 else 0.0
            
            job["domain_score"] = domain_score
            job["vector_score"] = vector_score
            job["combined_score"] = (domain_weight * domain_score) + \
                                   ((1 - domain_weight) * normalized_vector_score) + \
                                   domain_bonus
            
            if domain_score >= 0.7:
                job["confidence"] = "High"
            elif domain_score >= 0.3:
                job["confidence"] = "Medium"
            else:
                job["confidence"] = "Low"
        
        ranked_jobs = sorted(jobs, key=lambda x: x["combined_score"], reverse=True)
        
        # Apply diversity control
        final_ranking = []
        domain_counts = {}
        max_per_domain = max(3, len(ranked_jobs) // 5)
        
        for job in ranked_jobs:
            industry = job.get("industry", "Unknown")
            domain_counts[industry] = domain_counts.get(industry, 0)
            
            if domain_counts[industry] < max_per_domain:
                final_ranking.append(job)
                domain_counts[industry] += 1
            else:
                job["combined_score"] *= (1 - diversity_penalty)
                final_ranking.append(job)
        
        return sorted(final_ranking, key=lambda x: x["combined_score"], reverse=True)[:100]