import streamlit as st
from app import JobRecommenderApp

def main():
    """Main function to run the application."""
    app = JobRecommenderApp()
    try:
        app.run()
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()