#!/usr/bin/env python3
"""
Hybrid Dashboard - AP Policy Co-Pilot
Unified Streamlit interface supporting all processing modes
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
    page_title="AP Policy Co-Pilot - Hybrid",
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
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c, #d62728);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .hybrid-badge {
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .mode-card {
        background: #f8f9fa;
        border-left: 5px solid #007bff;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .citation-first-card {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #f8fff8, #e8f5e8);
    }
    
    .exploratory-card {
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #fffdf8, #faf8e8);
    }
    
    .balanced-card {
        border-left-color: #6f42c1;
        background: linear-gradient(135deg, #faf8ff, #f0e8ff);
    }
    
    .result-container {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .citation-card {
        background: #f0f9ff;
        border: 1px solid #0284c7;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .sota-result {
        background: #fef7cd;
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .confidence-high { color: #16a34a; font-weight: bold; }
    .confidence-medium { color: #eab308; font-weight: bold; }
    .confidence-low { color: #dc2626; font-weight: bold; }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .warning-banner {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .validation-pass {
        background: #dcfce7;
        border: 1px solid #16a34a;
        color: #166534;
    }
    
    .validation-fail {
        background: #fecaca;
        border: 1px solid #dc2626;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

def check_api_status():
    """Check if the hybrid API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def get_available_modes():
    """Get available processing modes from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/modes", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {
        "modes": ["auto", "citation_first", "exploratory", "balanced"],
        "descriptions": {
            "auto": "Intelligent mode selection",
            "citation_first": "Zero hallucination",
            "exploratory": "Cross-dataset insights",
            "balanced": "Best of both worlds"
        }
    }

def query_hybrid_api(query: str, mode: str = "auto", limit: int = 10, include_reasoning: bool = True):
    """Call the hybrid API"""
    try:
        payload = {
            "query": query,
            "mode": mode,
            "limit": limit,
            "include_reasoning": include_reasoning
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=45  # Longer timeout for hybrid processing
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.Timeout:
        return False, "Request timeout. Complex hybrid processing takes time."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API. Please ensure the hybrid server is running."
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

def render_mode_explanation(mode: str):
    """Render mode-specific explanation"""
    explanations = {
        "citation_first": {
            "icon": "📋",
            "title": "Citation-First Mode",
            "description": "Zero hallucination guarantee. Every claim is verified against sources.",
            "use_case": "Official policy queries, legal compliance, government use",
            "features": ["Mandatory citations", "Legal hierarchy validation", "Temperature 0.0"],
            "css_class": "citation-first-card"
        },
        "exploratory": {
            "icon": "🌉",
            "title": "Exploratory Mode", 
            "description": "Advanced insights using bridge tables and cross-dataset connections.",
            "use_case": "Research, trend analysis, comparative studies",
            "features": ["Bridge table connections", "Advanced GraphRAG", "Multi-agent reasoning"],
            "css_class": "exploratory-card"
        },
        "balanced": {
            "icon": "⚖️",
            "title": "Balanced Mode",
            "description": "Combines SOTA intelligence with citation validation.",
            "use_case": "Comprehensive analysis with accuracy validation",
            "features": ["SOTA + Citation validation", "Merged results", "Best of both worlds"],
            "css_class": "balanced-card"
        },
        "auto": {
            "icon": "🎯",
            "title": "Auto Mode",
            "description": "Intelligently selects the best mode based on query type.",
            "use_case": "General queries when unsure which mode to use",
            "features": ["Smart mode detection", "Keyword analysis", "Optimal processing"],
            "css_class": "mode-card"
        }
    }
    
    if mode not in explanations:
        return
    
    exp = explanations[mode]
    
    st.markdown(f"""
    <div class="mode-card {exp['css_class']}">
        <h3>{exp['icon']} {exp['title']}</h3>
        <p><strong>Description:</strong> {exp['description']}</p>
        <p><strong>Best for:</strong> {exp['use_case']}</p>
        <p><strong>Features:</strong> {', '.join(exp['features'])}</p>
    </div>
    """, unsafe_allow_html=True)

def render_citation_first_results(results: Dict[str, Any]):
    """Render citation-first specific results"""
    st.markdown("### 📋 Citation-First Results")
    
    # Main answer
    answer = results.get('answer', 'No response generated')
    st.markdown("#### 🤖 Response")
    st.write(answer)
    
    # Confidence and warnings
    confidence = results.get('confidence', 0.0)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**Confidence:** {render_confidence_badge(confidence)}", unsafe_allow_html=True)
    with col2:
        warnings = results.get('warnings', [])
        if warnings:
            st.warning(f"⚠️ {len(warnings)} warnings found")
    
    # Legal chain
    legal_chain = results.get('legal_chain', [])
    if legal_chain:
        st.markdown("#### 🏛️ Legal Hierarchy")
        st.write(" → ".join(legal_chain))
    
    # Citations
    citations = results.get('citations', [])
    if citations:
        st.markdown(f"#### 📚 Citations ({len(citations)})")
        for i, citation in enumerate(citations):
            with st.expander(f"Citation {i+1}: {citation.get('doc_title', 'Unknown')}"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**Type:** {citation.get('doc_type', 'Unknown')}")
                    st.write(f"**Excerpt:** {citation.get('excerpt', 'No excerpt')}")
                    if citation.get('section'):
                        st.write(f"**Section:** {citation.get('section')}")
                with col2:
                    st.write(f"**Confidence:** {citation.get('confidence', 0):.1%}")
                    if citation.get('issued_date'):
                        st.write(f"**Date:** {citation.get('issued_date')}")
                    if citation.get('supersedes'):
                        st.write(f"**Supersedes:** {', '.join(citation.get('supersedes', []))}")

def render_exploratory_results(results: Dict[str, Any]):
    """Render exploratory (SOTA) specific results"""
    st.markdown("### 🌉 Exploratory Results")
    
    # Results overview
    search_results = results.get('results', [])
    confidence = results.get('confidence', 0.0)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Results Found", len(search_results))
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
    with col3:
        bridge_connections = results.get('bridge_connections', False)
        st.metric("Bridge Tables", "✅ Active" if bridge_connections else "❌ Inactive")
    
    # Reasoning chain
    reasoning = results.get('reasoning', [])
    if reasoning:
        with st.expander("🧠 Reasoning Chain"):
            for i, step in enumerate(reasoning):
                st.write(f"**Step {i+1}:** {step}")
    
    # Search results
    if search_results:
        st.markdown("#### 📊 Search Results")
        for i, result in enumerate(search_results):
            with st.container():
                st.markdown(f"""
                <div class="sota-result">
                    <h5>🔍 Result {i + 1}</h5>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    content = result.get('content', result.get('text', 'No content'))
                    st.write(content[:300] + "..." if len(content) > 300 else content)
                with col2:
                    st.write(f"**Source:** {result.get('source_document', 'Unknown')}")
                    st.write(f"**Score:** {result.get('score', result.get('confidence', 0)):.2f}")

def render_balanced_results(results: Dict[str, Any]):
    """Render balanced mode results"""
    st.markdown("### ⚖️ Balanced Results")
    
    # Validation status
    validation_passed = results.get('validation_passed', False)
    validation_class = "validation-pass" if validation_passed else "validation-fail"
    validation_text = "✅ Validation Passed" if validation_passed else "❌ Validation Failed"
    
    st.markdown(f"""
    <div class="warning-banner {validation_class}">
        <strong>Citation Validation:</strong> {validation_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Main response (from citation-first)
    answer = results.get('answer', '')
    if answer:
        st.markdown("#### 🤖 Validated Response")
        st.write(answer)
    
    # Citations
    citations = results.get('citations', [])
    if citations:
        st.markdown(f"#### 📚 Validated Citations ({len(citations)})")
        for i, citation in enumerate(citations):
            st.markdown(f"""
            <div class="citation-card">
                <strong>Citation {i+1}:</strong> {citation.get('doc_title', 'Unknown')}<br>
                <em>{citation.get('excerpt', 'No excerpt')[:200]}...</em>
            </div>
            """, unsafe_allow_html=True)
    
    # SOTA results for additional context
    sota_results = results.get('sota_results', [])
    if sota_results:
        with st.expander(f"🌉 Additional SOTA Insights ({len(sota_results)} results)"):
            for i, result in enumerate(sota_results):
                col1, col2 = st.columns([3, 1])
                with col1:
                    content = result.get('content', result.get('text', 'No content'))
                    st.write(f"**{i+1}.** {content[:200]}...")
                with col2:
                    st.write(f"Score: {result.get('score', 0):.2f}")

def render_results(results: Dict[str, Any]):
    """Render results based on processing mode"""
    metadata = results.get('metadata', {})
    mode_used = metadata.get('mode_used', 'unknown')
    
    # Mode explanation
    render_mode_explanation(mode_used)
    
    # Mode-specific rendering
    if mode_used == "citation_first":
        render_citation_first_results(results)
    elif mode_used == "exploratory":
        render_exploratory_results(results)
    elif mode_used == "balanced":
        render_balanced_results(results)
    else:
        st.error(f"Unknown mode: {mode_used}")
    
    # Performance metrics
    st.markdown("### 📈 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        execution_time = metadata.get('execution_time', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>⏱️ Time</h4>
            <h2>{execution_time:.2f}s</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_results = results.get('total_results', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Results</h4>
            <h2>{total_results}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Mode</h4>
            <h2>{mode_used.title()}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        citation_validation = metadata.get('citation_validation', False)
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔒 Validated</h4>
            <h2>{'Yes' if citation_validation else 'No'}</h2>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">🏛️ AP Policy Co-Pilot</h1>', unsafe_allow_html=True)
    st.markdown('<div class="hybrid-badge">Hybrid Citation-First + SOTA System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Hybrid Configuration")
        
        # Check API status
        api_status, api_info = check_api_status()
        if api_status:
            st.success("✅ Hybrid API: Connected")
            if api_info:
                st.write(f"**Version:** {api_info.get('version', 'Unknown')}")
                st.write(f"**Components:** {len(api_info.get('components', []))}")
                st.write(f"**Modes:** {len(api_info.get('modes', []))}")
        else:
            st.error("❌ Hybrid API: Disconnected")
            st.markdown("""
            **To start the hybrid server:**
            ```bash
            python hybrid_orchestrator.py --action api
            ```
            """)
            return
        
        # Mode selection
        modes_info = get_available_modes()
        mode = st.selectbox(
            "🎯 Processing Mode",
            modes_info.get("modes", ["auto"]),
            help="Select how to process your query"
        )
        
        # Show mode description
        descriptions = modes_info.get("descriptions", {})
        if mode in descriptions:
            st.info(f"**{mode.title()}:** {descriptions[mode]}")
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            limit = st.slider("📊 Max Results", 1, 50, 10)
            include_reasoning = st.checkbox("🧠 Include Reasoning", True)
            
        # System info
        with st.expander("ℹ️ System Information"):
            st.write("**Architecture:** Hybrid Citation-First + SOTA")
            st.write("**Components:** Weaviate + Neo4j + Bridge Tables")
            st.write("**Features:** Anti-hallucination + Cross-dataset insights")
    
    # Main interface
    st.markdown("## 🔍 Hybrid Search Interface")
    
    # Initialize session state for selected example
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = ""
    
    # Mode examples (before text input)
    st.markdown("### 💡 Mode-Specific Examples")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Citation-First", "🌉 Exploratory", "⚖️ Balanced", "🎯 Auto"])
    
    with tab1:
        st.markdown("**Best for official/legal queries:**")
        examples = [
            "What does GO 456 state about teacher recruitment?",
            "Which act governs primary education admission?",
            "What are the legal requirements for school infrastructure?"
        ]
        for example in examples:
            if st.button(f"📝 {example}", key=f"citation_{example[:20]}"):
                st.session_state.selected_example = example
                st.rerun()
    
    with tab2:
        st.markdown("**Best for research and trends:**")
        examples = [
            "Compare enrollment trends across all districts",
            "How has infrastructure improved over 5 years?",
            "Show bridge connections between policy and outcomes"
        ]
        for example in examples:
            if st.button(f"📈 {example}", key=f"exploratory_{example[:20]}"):
                st.session_state.selected_example = example
                st.rerun()
    
    with tab3:
        st.markdown("**Best for comprehensive analysis:**")
        examples = [
            "What are the enrollment statistics with policy context?",
            "Teacher recruitment data with legal framework",
            "Infrastructure development with compliance verification"
        ]
        for example in examples:
            if st.button(f"⚖️ {example}", key=f"balanced_{example[:20]}"):
                st.session_state.selected_example = example
                st.rerun()
    
    with tab4:
        st.markdown("**Let the system choose the best mode:**")
        examples = [
            "Tell me about education policy in Anantapur",
            "How does teacher recruitment work?",
            "What are the latest policy changes?"
        ]
        for example in examples:
            if st.button(f"🎯 {example}", key=f"auto_{example[:20]}"):
                st.session_state.selected_example = example
                st.rerun()
    
    # Search input (after examples to use selected_example)
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your query about AP education policies:",
            value=st.session_state.selected_example,
            placeholder="e.g., What does GO 123 state about teacher recruitment in Krishna district?",
            key="search_query"
        )
        # Clear selected example after using it
        if st.session_state.selected_example:
            st.session_state.selected_example = ""
    
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    
    # Perform search only when button is clicked
    if search_button:
        if not query:
            st.warning("⚠️ Please enter a search query.")
            return
        
        with st.spinner(f"🔍 Processing query using {mode} mode..."):
            success, results = query_hybrid_api(query, mode, limit, include_reasoning)
        
        if success:
            # Display results
            render_results(results)
            
        else:
            st.error(f"Search failed: {results}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏛️ AP Education Policy Co-Pilot | Hybrid Architecture v3.0</p>
        <p>Citation-First Accuracy + SOTA Intelligence + Bridge Table Insights</p>
        <p>Powered by Weaviate + Neo4j + Advanced RAG + Anti-Hallucination Validation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()