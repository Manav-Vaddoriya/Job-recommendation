import streamlit as st
import weaviate
from weaviate.classes.query import MetadataQuery
from typing import List, Dict

class JobSearchClient:
    """Handles vector search operations with Weaviate."""
    
    def __init__(self, host: str = "localhost", port: int = 8090, grpc_port: int = 50051):
        self.client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=grpc_port
        )
    
    def vector_search_jobs(
        self,
        user_vector: List[float],
        query_text: str = "",
        limit: int = 200,
        alpha: float = 0.7
    ) -> List[Dict]:
        """Perform broad vector-based search to get diverse candidates."""
        try:
            jobs_collection = self.client.collections.get("JobCollection")
            
            response = jobs_collection.query.hybrid(
                query=query_text,
                vector=user_vector,
                alpha=alpha,
                limit=limit,
                return_metadata=MetadataQuery(score=True)
            )

            jobs = []
            for o in response.objects:
                props = o.properties
                jobs.append({
                    "title": props.get("title", "N/A"),
                    "industry": props.get("industry", "Unknown"),
                    "description": props.get("description", ""),
                    "company_id": props.get("company_id", "N/A"),
                    "score": o.metadata.score,
                    "vector_score": o.metadata.score,
                })
            
            return jobs

        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def close(self):
        """Close the Weaviate client connection."""
        self.client.close()