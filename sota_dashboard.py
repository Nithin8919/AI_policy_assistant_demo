#!/usr/bin/env python3
"""
SOTA Unified Dashboard - AP Policy Co-Pilot
Single comprehensive Streamlit frontend for all search methods
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AP Policy Co-Pilot - SOTA",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sota-badge {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .method-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .result-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def check_api_status():
    """Check if the API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def get_available_methods():
    """Get available search methods from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/methods", timeout=5)
        if response.status_code == 200:
            return response.json().get("methods", [])
    except:
        pass
    return ["auto", "vector", "hybrid", "graph", "bridge", "advanced"]

def search_api(query: str, method: str = "auto", limit: int = 10, include_reasoning: bool = True):
    """Call the search API"""
    try:
        payload = {
            "query": query,
            "method": method,
            "limit": limit,
            "include_reasoning": include_reasoning
        }
        
        response = requests.post(
            f"{API_BASE_URL}/search",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.Timeout:
        return False, "Request timeout. The query might be too complex."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API. Please ensure the server is running."
    except Exception as e:
        return False, f"Error: {str(e)}"

def render_confidence_badge(confidence: float):
    """Render confidence score with color coding"""
    if confidence >= 0.8:
        return f'<span class="confidence-high">🟢 {confidence:.1%}</span>'
    elif confidence >= 0.6:
        return f'<span class="confidence-medium">🟡 {confidence:.1%}</span>'
    else:
        return f'<span class="confidence-low">🔴 {confidence:.1%}</span>'

def render_result_card(result: Dict[str, Any], index: int):
    """Render a result card"""
    with st.container():
        st.markdown(f"""
        <div class="result-card">
            <h4>📄 Result {index + 1}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Content
            content = result.get('content', result.get('text', 'No content available'))
            st.write(content[:500] + "..." if len(content) > 500 else content)
        
        with col2:
            # Metadata
            st.write("**Source:**")
            st.write(result.get('source_document', 'Unknown'))
            st.write("**District:**")
            st.write(result.get('district', 'N/A'))
        
        with col3:
            # Confidence
            confidence = result.get('confidence', result.get('score', 0.0))
            st.markdown("**Confidence:**")
            st.markdown(render_confidence_badge(confidence), unsafe_allow_html=True)
            
            # Year
            year = result.get('year', result.get('data_year', 'N/A'))
            st.write(f"**Year:** {year}")

def render_search_results(results: Dict[str, Any]):
    """Render search results based on method used"""
    method = results.get('metadata', {}).get('method_used', results.get('method', 'unknown'))
    
    st.markdown(f"""
    <div class="method-card">
        <h3>🔍 Method Used: {method.title()}</h3>
        <p>Execution Time: {results.get('metadata', {}).get('execution_time', 0):.2f}s</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Handle different result formats
    if method == "gemini":
        render_gemini_results(results)
    elif method == "advanced":
        render_advanced_results(results)
    else:
        render_standard_results(results)

def render_gemini_results(results: Dict[str, Any]):
    """Render Gemini-specific results with citations"""
    st.markdown("### 🤖 AI-Generated Response")
    
    # Main answer
    answer = results.get('answer', 'No response generated')
    st.markdown(f"**Answer:** {answer}")
    
    # Confidence
    confidence = results.get('confidence', 0.0)
    st.markdown(f"**Confidence:** {render_confidence_badge(confidence)}", unsafe_allow_html=True)
    
    # Citations
    citations = results.get('citations', [])
    if citations:
        st.markdown("### 📚 Citations")
        for i, citation in enumerate(citations):
            with st.expander(f"Citation {i+1}: {citation.get('source_document', 'Unknown')}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Excerpt:** {citation.get('excerpt', 'No excerpt')}")
                with col2:
                    st.write(f"**District:** {citation.get('district', 'N/A')}")
                    st.write(f"**Year:** {citation.get('year', 'N/A')}")
                    st.write(f"**Confidence:** {citation.get('confidence_score', 0):.1%}")

def render_advanced_results(results: Dict[str, Any]):
    """Render advanced GraphRAG results with reasoning"""
    st.markdown("### 🧠 Advanced Multi-Agent Analysis")
    
    # Main results
    main_results = results.get('results', [])
    confidence = results.get('confidence', 0.0)
    
    st.markdown(f"**Overall Confidence:** {render_confidence_badge(confidence)}", unsafe_allow_html=True)
    
    # Reasoning chain (if available)
    reasoning = results.get('reasoning', [])
    if reasoning:
        with st.expander("🔗 Reasoning Chain"):
            for i, step in enumerate(reasoning):
                st.write(f"**Step {i+1}:** {step}")
    
    # Results
    if main_results:
        st.markdown("### 📊 Search Results")
        for i, result in enumerate(main_results):
            render_result_card(result, i)

def render_standard_results(results: Dict[str, Any]):
    """Render standard search results"""
    search_results = results.get('results', [])
    total_results = results.get('total_results', len(search_results))
    
    st.markdown(f"### 📊 Search Results ({total_results} found)")
    
    if search_results:
        for i, result in enumerate(search_results):
            render_result_card(result, i)
    else:
        st.info("No results found for your query.")

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">🏛️ AP Policy Co-Pilot</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sota-badge">State-of-the-Art RAG System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Search Configuration")
        
        # Check API status
        api_status, api_info = check_api_status()
        if api_status:
            st.success("✅ API Server: Connected")
            if api_info:
                st.write(f"**Version:** {api_info.get('version', 'Unknown')}")
                st.write(f"**Components:** {len(api_info.get('components', []))}")
        else:
            st.error("❌ API Server: Disconnected")
            st.markdown("""
            **To start the server:**
            ```bash
            python sota_orchestrator.py --action api
            ```
            """)
            return
        
        # Search method selection
        available_methods = get_available_methods()
        method = st.selectbox(
            "🎯 Search Method",
            available_methods,
            help="""
            - **auto**: Intelligent method selection
            - **vector**: Pure semantic search
            - **hybrid**: Vector + keyword search
            - **graph**: Knowledge graph reasoning
            - **bridge**: Bridge table connections
            - **advanced**: Multi-agent GraphRAG
            - **gemini**: AI-powered responses
            """
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            limit = st.slider("📊 Max Results", 1, 50, 10)
            include_reasoning = st.checkbox("🧠 Include Reasoning", True)
            
        # Method descriptions
        method_descriptions = {
            "auto": "🎯 Automatically selects the best method for your query",
            "vector": "🔍 Pure semantic similarity search using embeddings",
            "hybrid": "⚡ Combines vector search with keyword matching",
            "graph": "🕸️ Uses knowledge graph relationships for reasoning",
            "bridge": "🌉 Leverages cross-dataset connections",
            "advanced": "🧠 Multi-agent GraphRAG with reasoning chain",
            "gemini": "🤖 AI-powered responses with citations"
        }
        
        if method in method_descriptions:
            st.info(method_descriptions[method])
    
    # Main interface
    st.markdown("## 🔍 Search Interface")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your query about AP education policies:",
            placeholder="e.g., What are the enrollment statistics for Krishna district in 2023?",
            key="search_query"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    
    # Example queries
    st.markdown("### 💡 Example Queries")
    examples = [
        "What are the enrollment statistics for Krishna district?",
        "Compare teacher recruitment between 2022 and 2023",
        "How has infrastructure improved in rural schools?",
        "Show me the latest policy changes for primary education",
        "What is the dropout rate trend over the last 5 years?"
    ]
    
    example_cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(f"📝 {example[:30]}...", key=f"example_{i}"):
                st.session_state.search_query = example
                query = example
    
    # Perform search
    if search_button or query:
        if not query:
            st.warning("Please enter a search query.")
            return
        
        with st.spinner(f"🔍 Searching using {method} method..."):
            success, results = search_api(query, method, limit, include_reasoning)
        
        if success:
            # Display results
            render_search_results(results)
            
            # Performance metrics
            metadata = results.get('metadata', {})
            if metadata:
                st.markdown("### 📈 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>⏱️ Time</h4>
                        <h2>{metadata.get('execution_time', 0):.2f}s</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>📊 Results</h4>
                        <h2>{results.get('total_results', 0)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    method_used = metadata.get('method_used', method)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 Method</h4>
                        <h2>{method_used.title()}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    confidence = results.get('confidence', 0.8)  # Default confidence
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎪 Confidence</h4>
                        <h2>{confidence:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
        else:
            st.error(f"Search failed: {results}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏛️ AP Education Policy Co-Pilot | State-of-the-Art RAG System</p>
        <p>Powered by Weaviate + Neo4j + Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()