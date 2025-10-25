import streamlit as st
import os
from components.resume_processor import ResumeProcessor
from components.domain_predictor import DomainPredictor
from components.job_search import JobSearchClient
from components.job_ranker import JobRanker
from components.job_display import JobDisplay

class JobRecommenderApp:
    """Main application class that orchestrates the job recommendation system."""
    
    def __init__(self):
        self.embedder = None
        self.resume_processor = None
        self.domain_predictor = None
        self.job_search_client = None
        self.job_ranker = None
        self.job_display = None
        
        self.setup_page()
        self.initialize_components()
    
    def setup_page(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(page_title="AI Job Recommender", page_icon="💼", layout="wide")
        st.title("AI-Powered Job Recommendations")
    
    @st.cache_resource
    def load_embedder(_self):
        from fastembed import TextEmbedding
        return TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
    
    def initialize_components(self):
        """Initialize all components of the application."""
        st.sidebar.info("Loading AI models...")
        
        # Load embedder
        self.embedder = self.load_embedder()
        
        # Initialize components
        self.resume_processor = ResumeProcessor(self.embedder)
        self.domain_predictor = DomainPredictor(
            model_path=os.path.join("trained_models", "final_job_domain_model.pth"),
            domain_embed_map_path="domain_embed_map.pkl"
        )
        self.job_search_client = JobSearchClient()
        self.job_ranker = JobRanker()
        self.job_display = JobDisplay()
        
        st.sidebar.success("Model loaded and ready!")
    
    def process_resume_and_recommend(self, uploaded_file):
        """Process resume and generate job recommendations."""
        with st.spinner("Analyzing your resume..."):
            # Process uploaded file
            text, temp_path = self.resume_processor.process_uploaded_file(uploaded_file)
            
            try:
                # Generate embedding
                embedding = self.resume_processor.generate_embedding(text)
                
                # Predict domains
                top_domains = self.domain_predictor.predict_topk(embedding, k=10)
                
                # Vector search
                initial_jobs = self.job_search_client.vector_search_jobs(
                    user_vector=embedding.tolist(),
                    query_text=text[:500],
                    limit=200,
                    alpha=0.7
                )

                # Domain-aware re-ranking
                if initial_jobs:
                    relevant_jobs = self.job_ranker.filter_by_top_domains(
                        initial_jobs, top_domains, min_domain_score=0.05
                    )
                    final_jobs = self.job_ranker.domain_aware_reranking(
                        jobs=relevant_jobs if relevant_jobs else initial_jobs,
                        top_domains=top_domains,
                        user_embedding=embedding,
                        domain_weight=0.6
                    )
                    
                    # Display recommended jobs
                    self.job_display.display_recommended_jobs(final_jobs[:10])
                else:
                    st.error("No jobs found in the database. Please try again later.")

            finally:
                # Cleanup temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    def run(self):
        """Run the main application."""
        st.sidebar.header("Upload Your Resume")
        uploaded_file = st.sidebar.file_uploader("Choose PDF or TXT file", type=["pdf", "txt"])
        
        if uploaded_file:
            self.process_resume_and_recommend(uploaded_file)
    
    def cleanup(self):
        """Clean up resources."""
        if self.job_search_client:
            self.job_search_client.close()