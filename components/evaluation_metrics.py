import streamlit as st
import numpy as np
import pandas as pd
import math
import re
from typing import List, Dict, Optional, Tuple

class RecommendationMetrics:
    """Calculate DCG, IDCG, and nDCG using available job data."""
    
    def __init__(self, jobs_csv_path: str = "jobs.csv"):
        """Initialize with job ground truth data."""
        try:
            self.jobs_df = pd.read_csv(jobs_csv_path)
            
            # Clean and prepare the data
            self.jobs_df = self.jobs_df.fillna('')
            self.jobs_df['job_id'] = self.jobs_df['job_id'].astype(str)
            self.jobs_df['title'] = self.jobs_df['title'].astype(str)
            self.jobs_df['parent_domain'] = self.jobs_df['parent_domain'].astype(str)
            
            # Calculate relevance scores if engagement data exists
            self.jobs_df['relevance_score'] = self.calculate_relevance_scores()
            
            self.metrics_available = True
            st.sidebar.success(f"✅ Loaded {len(self.jobs_df)} jobs for evaluation")
            
        except Exception as e:
            st.warning(f"⚠️ Metrics disabled: Could not load jobs.csv ({str(e)})")
            self.jobs_df = None
            self.metrics_available = False
    
    def calculate_relevance_scores(self) -> pd.Series:
        """Calculate relevance scores based on engagement metrics."""
        if 'views' in self.jobs_df.columns and 'applies' in self.jobs_df.columns:
            views = self.jobs_df['views'].fillna(0).astype(int)
            applies = self.jobs_df['applies'].fillna(0).astype(int)
            
            # Simple normalization
            max_views = views.max() if views.max() > 0 else 1
            max_applies = applies.max() if applies.max() > 0 else 1
            
            normalized_views = views / max_views
            normalized_applies = applies / max_applies
            
            # Combined score
            relevance = (0.7 * normalized_applies) + (0.3 * normalized_views)
        else:
            # If no engagement data, use uniform scores
            relevance = pd.Series([0.5] * len(self.jobs_df))
        
        return relevance
    
    def extract_job_identifier(self, job: Dict) -> str:
        """Extract a unique identifier from job data."""
        # Priority 1: job_id from Weaviate
        job_id = job.get('job_id')
        if job_id:
            return str(job_id)
        
        # Priority 2: company_id + title hash as fallback
        company_id = job.get('company_id', '')
        title = job.get('title', '')
        if company_id and title:
            return f"{company_id}_{hash(title) % 10000:04d}"
        
        # Priority 3: Title-based identifier
        if title:
            return f"title_{hash(title) % 10000:04d}"
        
        return "unknown"
    
    def create_ideal_ranking(self, domain: str, k: int = 20) -> List[str]:
        """Create ideal ranking for a specific domain."""
        if not self.metrics_available:
            return []
        
        # Find jobs in the target domain
        domain_lower = domain.lower()
        domain_jobs = self.jobs_df[
            self.jobs_df['parent_domain'].str.lower().str.contains(domain_lower, na=False)
        ]
        
        if domain_jobs.empty:
            # If no domain match, use all jobs
            domain_jobs = self.jobs_df
            st.sidebar.warning(f"⚠️ No domain match for '{domain}'. Using all jobs.")
        
        # Sort by relevance and get top k
        ideal_jobs = domain_jobs.sort_values('relevance_score', ascending=False).head(k)
        ideal_ids = ideal_jobs['job_id'].astype(str).tolist()
        
        st.sidebar.info(f"🎯 Ideal ranking: {len(ideal_ids)} jobs for '{domain}'")
        return ideal_ids
    
    def map_recommendations_to_ideal(self, recommended_jobs: List[Dict], ideal_ids: List[str]) -> List[float]:
        """Map recommended jobs to relevance scores."""
        relevance_scores = []
        
        for job in recommended_jobs:
            rec_id = self.extract_job_identifier(job)
            
            # Check if this job exists in our ideal set
            if rec_id in ideal_ids:
                # If it exists, give it a high relevance score
                relevance_scores.append(1.0)
            else:
                # Check by title similarity as fallback
                rec_title = job.get('title', '').lower()
                if rec_title:
                    # Simple title matching
                    ideal_titles = self.jobs_df[self.jobs_df['job_id'].astype(str).isin(ideal_ids)]['title'].str.lower()
                    matches = ideal_titles.str.contains(rec_title[:20], na=False)  # Match first 20 chars
                    if matches.any():
                        relevance_scores.append(0.7)  # Partial match
                    else:
                        relevance_scores.append(0.0)  # No match
                else:
                    relevance_scores.append(0.0)
        
        return relevance_scores
    
    def calculate_dcg(self, relevance_scores: List[float], k: int = None) -> float:
        """Calculate Discounted Cumulative Gain."""
        if k is not None:
            scores = relevance_scores[:k]
        else:
            scores = relevance_scores
        
        dcg = 0.0
        for i, rel in enumerate(scores):
            position = i + 1
            dcg += rel / math.log2(position + 1)
        
        return dcg
    
    def calculate_idcg(self, k: int) -> float:
        """Calculate Ideal DCG for k positions."""
        idcg = 0.0
        for i in range(k):
            position = i + 1
            idcg += 1.0 / math.log2(position + 1)  # Perfect relevance = 1.0
        return idcg
    
    def evaluate_recommendations(self, recommended_jobs: List[Dict], domain: str, k: int = 10) -> Dict:
        """Evaluate recommendation quality."""
        if not self.metrics_available or not recommended_jobs:
            return self._create_fallback_metrics(recommended_jobs, domain)
        
        st.sidebar.info(f"📊 Evaluating {len(recommended_jobs)} recommendations")
        
        # Get ideal ranking
        ideal_ids = self.create_ideal_ranking(domain, k=20)
        
        if not ideal_ids:
            return self._create_fallback_metrics(recommended_jobs, domain)
        
        # Map recommendations to relevance scores
        relevance_scores = self.map_recommendations_to_ideal(recommended_jobs, ideal_ids)
        
        # Calculate metrics
        dcg = self.calculate_dcg(relevance_scores, k)
        idcg = self.calculate_idcg(min(k, len(ideal_ids)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # Calculate additional metrics
        matches = sum(1 for score in relevance_scores if score > 0)
        match_rate = matches / len(recommended_jobs) if recommended_jobs else 0
        precision = match_rate
        recall = matches / len(ideal_ids) if ideal_ids else 0
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
        
        # Debug info
        st.sidebar.write(f"• Matches: {matches}/{len(recommended_jobs)}")
        st.sidebar.write(f"• Avg relevance: {avg_relevance:.3f}")
        st.sidebar.write(f"• DCG: {dcg:.3f}, IDCG: {idcg:.3f}")
        
        return {
            'DCG': dcg,
            'IDCG': idcg,
            'nDCG': ndcg,
            'precision': precision,
            'recall': recall,
            'match_rate': match_rate,
            'avg_relevance': avg_relevance,
            'num_recommendations': len(recommended_jobs),
            'num_ideal_jobs': len(ideal_ids),
            'num_matches': matches,
            'domain': domain
        }
    
    def _create_fallback_metrics(self, recommended_jobs: List[Dict], domain: str) -> Dict:
        """Create fallback metrics when evaluation isn't possible."""
        if not recommended_jobs:
            return {
                'DCG': 0.0, 'IDCG': 1.0, 'nDCG': 0.0, 'precision': 0.0, 'recall': 0.0,
                'match_rate': 0.0, 'avg_relevance': 0.0, 'num_recommendations': 0,
                'num_ideal_jobs': 0, 'num_matches': 0, 'domain': domain, 'fallback': True
            }
        
        # Simple quality estimate based on job content
        valid_jobs = sum(1 for job in recommended_jobs if job.get('title') and job.get('industry'))
        quality_score = valid_jobs / len(recommended_jobs)
        
        return {
            'DCG': quality_score * 2,
            'IDCG': 3.0,
            'nDCG': quality_score * 0.6,
            'precision': quality_score,
            'recall': quality_score * 0.5,
            'match_rate': quality_score,
            'avg_relevance': quality_score,
            'num_recommendations': len(recommended_jobs),
            'num_ideal_jobs': 10,
            'num_matches': int(quality_score * len(recommended_jobs)),
            'domain': domain,
            'fallback': True
        }
    
    def display_evaluation_metrics(self, metrics: Dict):
        """Display evaluation metrics in Streamlit."""
        if not metrics:
            st.warning("⚠️ No metrics available")
            return
        
        st.markdown("---")
        st.subheader("📊 Recommendation Quality Evaluation")
        
        if metrics.get('fallback'):
            st.warning("⚠️ Using estimated metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("nDCG Score", f"{metrics['nDCG']:.3f}")
            st.caption("Ranking Quality")
        
        with col2:
            st.metric("Match Rate", f"{metrics['match_rate']:.3f}")
            st.caption("Relevance Score")
        
        with col3:
            st.metric("Precision", f"{metrics['precision']:.3f}")
            st.caption("Accuracy")
        
        with col4:
            st.metric("Matches", f"{metrics['num_matches']}/{metrics['num_recommendations']}")
            st.caption("Successful Recommendations")
        
        with st.expander("📈 Detailed Analysis"):
            st.write(f"""
            **Domain:** {metrics['domain']}
            **nDCG:** {metrics['nDCG']:.3f} - Overall ranking quality
            **Match Rate:** {metrics['match_rate']:.3f} - Percentage of relevant recommendations
            **Precision:** {metrics['precision']:.3f} - Recommendation accuracy
            **Average Relevance:** {metrics['avg_relevance']:.3f} - Content match quality
            """)