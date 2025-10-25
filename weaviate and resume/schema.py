import json

def create_job_collection_schema():
    """Create schema for job collection with proper field definitions"""

    schema = {
        "class": "JobCollection",
        "description": "Collection for storing job postings with embeddings",
        "vectorizer": "none",  # We'll provide our own vectors
        "properties": [
            {
                "name": "job_id",
                "dataType": ["text"],
                "description": "Unique identifier for the job"
            },
            {
                "name": "company_id",
                "dataType": ["text"],
                "description": "Identifier for the company posting the job"
            },
            {
                "name": "title",
                "dataType": ["text"],
                "description": "Job title",
                "tokenization": "word"
            },
            {
                "name": "description",
                "dataType": ["text"],
                "description": "Full job description",
                "tokenization": "word"
            },
            {
                "name": "industry",
                "dataType": ["text"],
                "description": "Industry category of the job"
            }
        ]
    }

    return schema

# Create the schema
job_schema = create_job_collection_schema()
print("Job collection schema created:")
print(json.dumps(job_schema, indent=2))