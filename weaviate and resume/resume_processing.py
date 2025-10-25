import streamlit as st
st.set_page_config(page_title="Fast Resume Embedding (1024-D)", layout="wide")

from fastembed import TextEmbedding
import numpy as np
import time
from PyPDF2 import PdfReader
import tempfile
import os

@st.cache_resource
def load_model():
    return TextEmbedding(model_name="BAAI/bge-large-en-v1.5") 

# Load model *before* user uploads
st.sidebar.info("⚙️ Loading 1024-D embedding model...")
embedder = load_model()
st.sidebar.success("✅ Model loaded and ready!")
st.title("⚡ Fast Resume Embedding Generator (1024-D)")
st.write("Upload your resume to generate a 1024-dimensional embedding using **BAAI/bge-large-en-v1.5**")

uploaded_file = st.file_uploader("📄 Upload your resume", type=["pdf", "txt"])

if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(temp_path)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
    else:
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    st.info(f"Extracted {len(text)} characters from the resume")

    # Generate embedding
    with st.spinner("Generating 1024-dimensional embedding..."):
        start = time.time()
        embedding = np.array(list(embedder.embed([text]))[0])
        end = time.time()

    # Display results
    st.success("🎉 Embedding generated successfully!")
    st.write(f"**Embedding Dimension:** {len(embedding)}")
    st.write(f"**Embedding Shape:** {embedding.shape}")
    st.write(f"**Processing Time:** {end - start:.3f} seconds")

    # Show preview
    st.subheader("🧠 Embedding Preview (first 50 values)")
    st.write(embedding[:50])

    # Download option
    np.save("resume_embedding_1024.npy", embedding)
    with open("resume_embedding_1024.npy", "rb") as f:
        st.download_button(
            label="💾 Download Embedding (.npy)",
            data=f,
            file_name="resume_embedding_1024.npy",
            mime="application/octet-stream"
        )

    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
