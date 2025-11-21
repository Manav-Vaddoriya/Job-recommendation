import tempfile
import os
from PyPDF2 import PdfReader
import numpy as np
from typing import Tuple

class ResumeProcessor:
    """Handles resume text extraction and processing."""
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    
    def process_uploaded_file(self, uploaded_file) -> Tuple[str, str]:
        """Process uploaded file and return text and file path.

        Supports both file-like objects (Streamlit/FastAPI) and string paths.
        """
        # Case 1: file-like object (UploadFile, Streamlit upload, etc.)
        if hasattr(uploaded_file, "read"):
            file_name = getattr(uploaded_file, "filename", getattr(uploaded_file, "name", "uploaded_resume.pdf"))
            suffix = os.path.splitext(file_name)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

        # Case 2: string file path (already saved on disk)
        elif isinstance(uploaded_file, str) and os.path.exists(uploaded_file):
            temp_path = uploaded_file

        else:
            raise ValueError("Invalid uploaded_file: expected file object or valid file path")

        # Extract text based on file extension
        if temp_path.lower().endswith(".pdf"):
            text = self.extract_text_from_pdf(temp_path)
        else:
            text = self.extract_text_from_txt(temp_path)
            
        return text, temp_path
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding from text."""
        return np.array(list(self.embedder.embed([text]))[0])
