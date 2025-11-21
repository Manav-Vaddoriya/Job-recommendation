import os
import psutil
import time
import uuid
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable Streamlit for API mode
os.environ["STREAMLIT_SERVER_PORT"] = "0"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

app = FastAPI(
    title="AI Job Recommender API",
    description="API for job recommendations based on resume analysis",
    version="1.0.0"
)

# Global recommender instance
recommender = None

def initialize_recommender():
    """Initialize the recommender system with proper error handling"""
    global recommender
    try:
        from components.resume_processor import ResumeProcessor
        from components.domain_predictor import DomainPredictor
        from components.job_search import JobSearchClient
        from components.job_ranker import JobRanker
        
        from fastembed import TextEmbedding
        
        logger.info("Loading AI models...")

        embedder = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")

        resume_processor = ResumeProcessor(embedder)
        domain_predictor = DomainPredictor(
            model_path=os.path.join("trained_models", "final_job_domain_model.pth"),
            domain_embed_map_path="domain_embed_map.pkl"
        )
        job_search_client = JobSearchClient()
        job_ranker = JobRanker()

        class SimpleRecommender:
            def __init__(self):
                self.resume_processor = resume_processor
                self.domain_predictor = domain_predictor
                self.job_search_client = job_search_client
                self.job_ranker = job_ranker
        
        recommender = SimpleRecommender()
        logger.info("✅ All models loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize recommender: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize recommender on startup"""
    await asyncio.get_event_loop().run_in_executor(None, initialize_recommender)


# -----------------------------------------------------------
# ONLY /recommend ENDPOINT KEPT
# -----------------------------------------------------------
@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    """Main recommendation endpoint with performance monitoring"""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service initializing")

    start_time = time.time()
    process = psutil.Process(os.getpid())

    # Validate file type
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        with open(temp_path, "wb") as f:
            f.write(contents)

        logger.info(f"Processing file: {file.filename}")

        # Stage timings
        t0 = time.time()
        text, _ = recommender.resume_processor.process_uploaded_file(temp_path)
        t1 = time.time()

        embedding = recommender.resume_processor.generate_embedding(text)
        t2 = time.time()

        top_domains = recommender.domain_predictor.predict_topk(embedding, k=10)
        t3 = time.time()

        initial_jobs = recommender.job_search_client.vector_search_jobs(
            user_vector=embedding.tolist(),
            query_text=text[:500],
            limit=200,
            alpha=0.7
        )
        t4 = time.time()

        if initial_jobs:
            relevant_jobs = recommender.job_ranker.filter_by_top_domains(
                initial_jobs, top_domains, min_domain_score=0.05
            )
            final_jobs = recommender.job_ranker.domain_aware_reranking(
                jobs=relevant_jobs if relevant_jobs else initial_jobs,
                top_domains=top_domains,
                user_embedding=embedding,
                domain_weight=0.6
            )
        else:
            final_jobs = []

        t5 = time.time()

        final_jobs = final_jobs[:10]

        total_time = round((t5 - start_time) * 1000, 2)
        cpu_usage = psutil.cpu_percent(interval=0.1)
        mem_usage = process.memory_info().rss / (1024 * 1024)

        timing_breakdown = {
            "resume_processing_ms": round((t1 - t0) * 1000, 2),
            "embedding_generation_ms": round((t2 - t1) * 1000, 2),
            "domain_prediction_ms": round((t3 - t2) * 1000, 2),
            "vector_search_ms": round((t4 - t3) * 1000, 2),
            "reranking_ms": round((t5 - t4) * 1000, 2),
            "total_pipeline_ms": total_time
        }

        response_data = {
            "status": "success",
            "file_processed": file.filename,
            "num_recommendations": len(final_jobs),
            "recommendations": final_jobs,
            "cpu_usage_percent": cpu_usage,
            "memory_usage_mb": round(mem_usage, 2),
            "timing": timing_breakdown
        }

        logger.info(f"✅ Request completed in {total_time}ms ({len(final_jobs)} jobs)")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
