import streamlit as st
from typing import List, Dict

class JobDisplay:
    """Handles the display of job recommendations."""
    
    @staticmethod
    def display_recommended_jobs(jobs: List[Dict]):
        """Display only the recommended jobs in a clean, professional format."""
        if not jobs:
            st.warning("No job recommendations available. Try adjusting your resume or search criteria.")
            return
        
        st.markdown("---")
        
        # Header with summary
        st.subheader(f"Recommended Jobs ({len(jobs)} positions)")
        
        # Display each job recommendation
        for idx, job in enumerate(jobs, 1):
            score = job.get("combined_score", 0.0)
            domain_score = job.get("domain_score", 0.0)
            vector_score = job.get("vector_score", 0.0)
            confidence = job.get("confidence", "Unknown")
            
            # Determine confidence badge color
            if confidence == "High":
                badge_color = "🟢"
                match_text = "Excellent Match"
            elif confidence == "Medium":
                badge_color = "🟡" 
                match_text = "Good Match"
            else:
                badge_color = "🔴"
                match_text = "Potential Match"
            
            # Create job card
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"### {idx}. {job['title']}")
                    st.write(f"**Industry:** {job.get('industry', 'N/A')}")
                    
                    # Truncate description for clean display
                    description = job["description"]
                    if len(description) > 300:
                        description = description[:300] + "..."
                    st.write(description)
                    
                    st.write(f"**Company ID:** {job['company_id']}")
                
                with col2:
                    st.metric("Match Score", f"{score:.1%}")
                    st.write(f"{badge_color} **{match_text}**")
                    
                    # Mini metrics
                    st.caption(f"Domain Fit: {domain_score:.1%}")
                    st.caption(f"Content Match: {vector_score:.3f}")
                
                st.markdown("---")