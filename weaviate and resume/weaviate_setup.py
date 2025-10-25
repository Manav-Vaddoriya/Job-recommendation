import weaviate
from weaviate.classes.config import Property, DataType, Configure
import pandas as pd
import pickle

# =====================================================
# LOAD DATA
# =====================================================
print("Starting data loading process...")
df = pd.read_parquet("data/processed_data.parquet")
print("✅ Data loaded successfully")

# Load job_id mapping
with open("data/job_id_to_id_mapping.pkl", "rb") as f:
    id_mapping = pickle.load(f)
print("✅ Job ID mapping loaded")

# Replace job IDs with mapped IDs
df["job_id"] = df["job_id"].map(id_mapping)
print("✅ Job IDs mapped")

# Load job embeddings
with open("data/job_embed_dict.pkl", "rb") as f:
    embed_mapping = pickle.load(f)
print("✅ Embedding mapping loaded")

# Map embeddings to job IDs
df["job_embed"] = df["job_id"].map(embed_mapping)
print("✅ Job embeddings mapped")

# Create the final dataframe for insertion
jobs_df = df[["job_id", "company_id", "title", "description", "industry", "job_embed"]]
print("✅ Final DataFrame prepared for insertion")

# =====================================================
# CONNECT TO WEAVIATE
# =====================================================
client = weaviate.connect_to_local(
    host="localhost",
    port=8090,
    grpc_port=50051
)
print("🔗 Connected to Weaviate:", client.is_ready())

# =====================================================
# SETUP COLLECTION (IF NOT EXISTS)
# =====================================================
def setup_job_collection():
    """
    Create the JobCollection only if it doesn't exist.
    """
    if not client.collections.exists("JobCollection"):
        client.collections.create(
            name="JobCollection",
            description="Collection for storing job postings with embeddings",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="job_id", data_type=DataType.TEXT, description="Unique job identifier"),
                Property(name="company_id", data_type=DataType.TEXT, description="Company posting the job"),
                Property(name="title", data_type=DataType.TEXT, description="Job title"),
                Property(name="description", data_type=DataType.TEXT, description="Full job description"),
                Property(name="industry", data_type=DataType.TEXT, description="Industry category of the job"),
            ]
        )
        print("✨ JobCollection created successfully")
    else:
        print("✅ JobCollection already exists, skipping creation")

setup_job_collection()

# =====================================================
# INSERT DATA INTO WEAVIATE
# =====================================================
def insert_job_data(job_data, job_embeddings):
    """
    Insert job postings with embeddings into Weaviate (if not already inserted).
    """
    print(f"📦 Preparing to insert {len(job_data)} jobs...")

    jobs_collection = client.collections.get("JobCollection")
    count = jobs_collection.aggregate.over_all(total_count=True).total_count
    print(f"📊 Current objects in JobCollection: {count}")

    # Optional: Skip if data already exists
    if count > 0:
        print("⚠️ JobCollection already has data. Skipping insertion.")
        return

    with jobs_collection.batch.dynamic() as batch:
        for i, (job, embedding) in enumerate(zip(job_data, job_embeddings)):
            data_object = {
                "job_id": str(job.get("job_id", "")),
                "company_id": str(job.get("company_id", "")),
                "title": str(job.get("title", "")),
                "description": str(job.get("description", "")),
                "industry": str(job.get("industry", "")),
            }
            batch.add_object(
                properties=data_object,
                vector=embedding
            )
            if (i + 1) % 100 == 0:
                print(f"📤 Added {i + 1}/{len(job_data)} jobs to the batch")

    print("✅ Batch insertion completed successfully!")

# Prepare job data
job_data_list = jobs_df[["job_id", "company_id", "title", "description", "industry"]].to_dict(orient="records")
job_embeddings_list = jobs_df["job_embed"].tolist()

insert_job_data(job_data_list, job_embeddings_list)

# =====================================================
# CLOSE CONNECTION
# =====================================================
client.close()
print("🔒 Weaviate client closed. Database setup complete and persistent.")
