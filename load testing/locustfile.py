from locust import HttpUser, task, between, events
import os
import random
import json
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

class JobRecommenderUser(HttpUser):
    """
    Locust user class for testing ONLY the /recommend endpoint
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    host = "http://localhost:8000"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_files = self._get_sample_files()

    def _get_sample_files(self):
        """Load sample resumes"""
        sample_dir = "sample_resumes"
        files = []

        if os.path.exists(sample_dir):
            for file in os.listdir(sample_dir):
                if file.lower().endswith(('.pdf', '.txt')):
                    files.append(os.path.join(sample_dir, file))

        if not files:
            # Create dummy test file
            os.makedirs(sample_dir, exist_ok=True)
            dummy_file = os.path.join(sample_dir, "test_resume.txt")
            with open(dummy_file, "w") as f:
                f.write("Software Engineer with 5 years experience in Python, ML, and FastAPI.")
            files = [dummy_file]
            print(f"📝 Created dummy test file: {dummy_file}")

        print(f"📁 Found {len(files)} sample files for testing")
        return files

    def on_start(self):
        """Called on user start"""
        print(f"🚀 User started - Testing {self.host}")

    @task
    def test_recommendation(self):
        """Load test for the /recommend endpoint"""
        if not self.sample_files:
            print("❌ No sample files available")
            return

        resume_path = random.choice(self.sample_files)

        with open(resume_path, "rb") as file:
            files = {
                "file": (
                    os.path.basename(resume_path),
                    file,
                    "application/pdf" if resume_path.lower().endswith('.pdf') else "text/plain"
                )
            }

            start_time = time.time()

            with self.client.post(
                "/recommend",
                files=files,
                name="/recommend",
                catch_response=True,
                timeout=200  # Needs long timeout due to embedding + ML pipeline
            ) as response:

                request_time = (time.time() - start_time) * 1000  

                if response.status_code != 200:
                    response.failure(f"Status {response.status_code}: {response.text[:200]}")
                    return

                try:
                    data = response.json()
                except json.JSONDecodeError:
                    response.failure(f"Invalid JSON response: {response.text[:200]}")
                    return

                # Validate response
                if data.get("status") != "success":
                    response.failure(f"API Error: {data.get('message', 'Unknown error')}")
                    return

                # Success
                num_recommendations = data.get("num_recommendations", 0)
                total_time = data.get("timing", {}).get("total_pipeline_ms", 0)

                response.success()
                print(f"✅ Success: {total_time} ms | {num_recommendations} jobs")

# Test start event
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("🧪 Starting Load Test...")
    print(f"📊 Target Host: {environment.host}")

# Test stop event
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("🏁 Load Test Completed!")


# uvicorn api_app:app --host 0.0.0.0 --port 8000 --reload
# locust -f locustfile.py --host=http://localhost:8000
# curl.exe -X POST "http://localhost:8000/recommend" -F "file=@sample_resumes/HR.pdf"