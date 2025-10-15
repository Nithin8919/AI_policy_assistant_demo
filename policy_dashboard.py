#!/usr/bin/env python3
"""
Streamlit Dashboard for AP Policy Co-Pilot
Citation-first interface for policy reasoning
"""
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Page config
st.set_page_config(
    page_title="AP Policy Co-Pilot",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .citation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .legal-chain {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏛️ AP Policy Co-Pilot</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666;">Citation-First Policy Reasoning for Andhra Pradesh School Education</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API endpoint
    api_url = st.text_input(
        "API Endpoint",
        value="http://localhost:8000",
        help="Base URL of the Policy Co-Pilot API"
    )
    
    st.divider()
    
    # Query options
    st.subheader("Query Options")
    
    query_type = st.selectbox(
        "Query Type",
        options=["auto", "legal", "data", "combined"],
        help="auto: system decides, legal: Acts/Rules/GOs, data: statistics, combined: both"
    )
    
    max_results = st.slider(
        "Max Results",
        min_value=5,
        max_value=20,
        value=10,
        help="Maximum number of documents to retrieve"
    )
    
    require_citations = st.checkbox(
        "Require Citations",
        value=True,
        help="Reject responses without sufficient citations"
    )
    
    st.divider()
    
    # Test queries
    st.subheader("📝 Sample Queries")
    
    sample_queries = [
        "What are the responsibilities of School Management Committees?",
        "What was the dropout rate among ST students in 2016-17?",
        "Which GO governs Nadu-Nedu infrastructure?",
        "How many schools offered Telugu medium in 2016-17?",
        "What is the legal framework for mid-day meals and budget spent?"
    ]
    
    selected_sample = st.selectbox(
        "Select a sample query",
        options=[""] + sample_queries
    )
    
    if st.button("Use Sample Query") and selected_sample:
        st.session_state.query = selected_sample
    
    st.divider()
    
    # System status
    st.subheader("🏥 System Status")
    
    if st.button("Check Health"):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ System Operational")
            else:
                st.error(f"❌ System Error: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection Failed: {e}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Ask a Policy Question")
    
    # Query input
    query = st.text_area(
        "Your Question",
        value=st.session_state.get('query', ''),
        height=100,
        placeholder="Example: What was the dropout rate among SC students in Anantapur district in 2016-17?",
        key="query_input"
    )
    
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not query:
            st.warning("Please enter a question")
        else:
            with st.spinner("Searching policy documents..."):
                try:
                    # Call API
                    response = requests.post(
                        f"{api_url}/query",
                        json={
                            "query": query,
                            "query_type": query_type,
                            "require_citations": require_citations,
                            "max_results": max_results
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.last_result = result
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                        st.session_state.last_result = None
                        
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.last_result = None

with col2:
    st.header("📊 Quick Stats")
    
    # Display metrics if results available
    if 'last_result' in st.session_state and st.session_state.last_result:
        result = st.session_state.last_result
        
        st.metric("Confidence", f"{result.get('confidence_score', 0):.1%}")
        st.metric("Citations", len(result.get('citations', [])))
        st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
        
        # Legal chain
        if result.get('legal_chain'):
            st.subheader("⚖️ Legal Hierarchy")
            for item in result['legal_chain']:
                st.text(f"→ {item}")
    else:
        st.info("Submit a query to see statistics")

# Results section
if 'last_result' in st.session_state and st.session_state.last_result:
    result = st.session_state.last_result
    
    st.divider()
    
    # Answer section
    st.header("📄 Answer")
    
    answer_container = st.container()
    with answer_container:
        st.markdown(result.get('answer', 'No answer generated'))
    
    # Warnings
    if result.get('warnings'):
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ Validation Warnings")
        for warning in result['warnings']:
            st.write(f"- {warning}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Citations section
    st.header("📚 Citations")
    
    citations = result.get('citations', [])
    
    if citations:
        st.write(f"Found {len(citations)} supporting sources:")
        
        for i, citation in enumerate(citations, 1):
            with st.expander(f"Citation {i}: {citation.get('doc_title', 'Unknown Document')}", expanded=(i == 1)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Document:** {citation.get('doc_title', 'N/A')}")
                    st.markdown(f"**Type:** {citation.get('doc_type', 'N/A')}")
                    
                    if citation.get('section'):
                        st.markdown(f"**Section:** {citation.get('section')}")
                    
                    if citation.get('go_number'):
                        st.markdown(f"**GO Number:** {citation.get('go_number')}")
                    
                    if citation.get('issued_date'):
                        st.markdown(f"**Date:** {citation.get('issued_date')}")
                    
                    if citation.get('indicator'):
                        st.markdown(f"**Indicator:** {citation.get('indicator')}")
                    
                    if citation.get('district'):
                        st.markdown(f"**District:** {citation.get('district')}")
                    
                    if citation.get('year'):
                        st.markdown(f"**Year:** {citation.get('year')}")
                
                with col2:
                    st.metric("Confidence", f"{citation.get('confidence', 0):.1%}")
                    
                    if citation.get('page_number'):
                        st.write(f"📄 Page {citation.get('page_number')}")
                
                # Excerpt
                if citation.get('excerpt'):
                    st.markdown("**Excerpt:**")
                    st.markdown(f'<div class="citation-box">{citation.get("excerpt")}</div>', unsafe_allow_html=True)
                
                # Source URL
                if citation.get('source_url'):
                    st.markdown(f"🔗 [View Source]({citation['source_url']})")
    else:
        st.warning("No citations available for this response")
    
    st.divider()
    
    # Data points section
    if result.get('data_points'):
        st.header("📈 Data Points")
        
        data_df = pd.DataFrame(result['data_points'])
        
        # Display as table
        st.dataframe(
            data_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Visualization if applicable
        if 'district' in data_df.columns and 'value' in data_df.columns:
            st.subheader("Visualization")
            
            chart_type = st.selectbox(
                "Chart Type",
                options=["Bar Chart", "Line Chart", "Table Only"]
            )
            
            if chart_type == "Bar Chart":
                st.bar_chart(data_df.set_index('district')['value'])
            elif chart_type == "Line Chart" and 'year' in data_df.columns:
                st.line_chart(data_df.set_index('year')['value'])
    
    st.divider()
    
    # Export options
    st.header("💾 Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as JSON
        json_data = json.dumps(result, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"policy_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export as Markdown
        markdown = f"# Query: {result['query']}\n\n"
        markdown += f"## Answer\n{result['answer']}\n\n"
        markdown += f"## Legal Chain\n"
        for item in result.get('legal_chain', []):
            markdown += f"- {item}\n"
        markdown += f"\n## Citations\n"
        for i, citation in enumerate(citations, 1):
            markdown += f"{i}. {citation.get('doc_title', 'Unknown')}\n"
        
        st.download_button(
            label="Download Markdown",
            data=markdown,
            file_name=f"policy_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    with col3:
        # Export citations as CSV
        if citations:
            citations_df = pd.DataFrame(citations)
            csv = citations_df.to_csv(index=False)
            st.download_button(
                label="Download Citations CSV",
                data=csv,
                file_name=f"citations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.divider()
st.markdown(
    '<p style="text-align: center; color: #999; font-size: 0.9rem;">'
    '🏛️ AP Policy Co-Pilot | Citation-First Policy Reasoning | Powered by RAG + Knowledge Graph'
    '</p>',
    unsafe_allow_html=True
)