#!/usr/bin/env python3
"""
Stage 7: Streamlit Dashboard
Interactive dashboard for AP education policy intelligence
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Set page config
st.set_page_config(
    page_title="AP Education Policy Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

logger = logging.getLogger(__name__)

class APEducationDashboard:
    """Production-ready Streamlit dashboard for AP education policy intelligence"""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.indicators = [
            'GER', 'NER', 'GPI', 'PTR', 'Dropout_Rate', 'Retention_Rate',
            'Enrolment', 'Teachers', 'Schools', 'Classrooms'
        ]
        self.districts = [
            'Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Kadapa',
            'Krishna', 'Kurnool', 'Nellore', 'Prakasam', 'Srikakulam',
            'Visakhapatnam', 'Vizianagaram', 'West Godavari'
        ]
        self.years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']
    
    def make_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make API request to RAG server"""
        try:
            url = f"{self.api_base_url}{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return {}
    
    def post_api_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST API request to RAG server"""
        try:
            url = f"{self.api_base_url}{endpoint}"
            response = requests.post(url, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return {}
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">📊 Andhra Pradesh Education Policy Intelligence</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with filters and controls"""
        st.sidebar.markdown('<div class="sidebar-header">🔍 Search & Filters</div>', 
                           unsafe_allow_html=True)
        
        # Search query
        search_query = st.sidebar.text_input(
            "Search Query",
            placeholder="e.g., GER in Visakhapatnam",
            help="Enter your search query here"
        )
        
        # Filters
        st.sidebar.markdown("**Filters**")
        
        selected_indicator = st.sidebar.selectbox(
            "Indicator",
            ["All"] + self.indicators,
            help="Select education indicator"
        )
        
        selected_district = st.sidebar.selectbox(
            "District",
            ["All"] + self.districts,
            help="Select district"
        )
        
        selected_year = st.sidebar.selectbox(
            "Year",
            ["All"] + self.years,
            help="Select year"
        )
        
        # Search options
        st.sidebar.markdown("**Search Options**")
        
        include_vector = st.sidebar.checkbox("Vector Search", value=True)
        include_graph = st.sidebar.checkbox("Graph Search", value=True)
        
        limit = st.sidebar.slider("Results Limit", 5, 50, 10)
        
        # Build filters dict
        filters = {}
        if selected_indicator != "All":
            filters["indicator"] = selected_indicator
        if selected_district != "All":
            filters["district"] = selected_district
        if selected_year != "All":
            filters["year"] = selected_year
        
        return {
            "query": search_query,
            "filters": filters,
            "include_vector": include_vector,
            "include_graph": include_graph,
            "limit": limit
        }
    
    def render_search_results(self, search_params: Dict[str, Any]):
        """Render search results"""
        if not search_params["query"]:
            st.info("Enter a search query to see results")
            return
        
        with st.spinner("Searching..."):
            search_data = {
                "query": search_params["query"],
                "limit": search_params["limit"],
                "include_graph": search_params["include_graph"],
                "include_vector": search_params["include_vector"],
                "filters": search_params["filters"]
            }
            
            results = self.post_api_request("/search", search_data)
        
        if not results:
            st.error("Search failed. Please check your query and try again.")
            return
        
        st.markdown("### 🔍 Search Results")
        
        # Display search metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Results", results.get("total_count", 0))
        with col2:
            st.metric("Query Time", f"{results.get('query_time', 0):.3f}s")
        with col3:
            st.metric("Search Method", results.get("search_method", "unknown"))
        with col4:
            st.metric("Query", search_params["query"][:30] + "...")
        
        # Display results
        search_results = results.get("results", [])
        if search_results:
            # Convert to DataFrame for better display
            df = pd.DataFrame(search_results)
            
            # Display table
            st.markdown("#### 📋 Results Table")
            st.dataframe(df, use_container_width=True)
            
            # Display individual results
            st.markdown("#### 📊 Detailed Results")
            for i, result in enumerate(search_results):
                with st.expander(f"Result {i+1}: {result.get('indicator', 'Unknown')} - {result.get('district', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Indicator:** {result.get('indicator', 'N/A')}")
                        st.markdown(f"**District:** {result.get('district', 'N/A')}")
                        st.markdown(f"**Year:** {result.get('year', 'N/A')}")
                        st.markdown(f"**Value:** {result.get('value', 'N/A')}")
                    
                    with col2:
                        st.markdown(f"**Unit:** {result.get('unit', 'N/A')}")
                        st.markdown(f"**Source:** {result.get('source', 'N/A')}")
                        st.markdown(f"**Confidence:** {result.get('confidence', 'N/A')}")
                        if 'similarity_score' in result:
                            st.markdown(f"**Similarity:** {result['similarity_score']:.3f}")
                    
                    if 'span_text' in result:
                        st.markdown(f"**Description:** {result['span_text']}")
        else:
            st.info("No results found for your query.")
    
    def render_trend_analysis(self):
        """Render trend analysis section"""
        st.markdown("### 📈 Trend Analysis")
        
        selected_indicator = st.selectbox(
            "Select Indicator for Trend Analysis",
            self.indicators,
            key="trend_indicator"
        )
        
        if st.button("Analyze Trends", key="trend_button"):
            with st.spinner("Analyzing trends..."):
                trends = self.make_api_request(f"/trends/{selected_indicator}")
            
            if trends and "trends" in trends:
                trend_data = trends["trends"]
                if trend_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(trend_data)
                    
                    # Create trend chart
                    fig = px.line(
                        df, 
                        x='year', 
                        y='value', 
                        color='district',
                        title=f'{selected_indicator} Trends by District',
                        labels={'value': selected_indicator, 'year': 'Year'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display trend table
                    st.markdown("#### 📋 Trend Data")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info(f"No trend data available for {selected_indicator}")
            else:
                st.error("Trend analysis failed")
    
    def render_comparison_analysis(self):
        """Render comparison analysis section"""
        st.markdown("### 🏆 District Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_indicator = st.selectbox(
                "Select Indicator",
                self.indicators,
                key="comp_indicator"
            )
        
        with col2:
            selected_year = st.selectbox(
                "Select Year",
                self.years,
                key="comp_year"
            )
        
        if st.button("Compare Districts", key="comp_button"):
            with st.spinner("Comparing districts..."):
                comparison = self.make_api_request(f"/compare/{selected_indicator}/{selected_year}")
            
            if comparison and "comparison" in comparison:
                comp_data = comparison["comparison"]
                if comp_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(comp_data)
                    
                    # Create comparison chart
                    fig = px.bar(
                        df,
                        x='district',
                        y='value',
                        title=f'{selected_indicator} by District ({selected_year})',
                        labels={'value': selected_indicator, 'district': 'District'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display comparison table
                    st.markdown("#### 📋 Comparison Data")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info(f"No comparison data available for {selected_indicator} in {selected_year}")
            else:
                st.error("Comparison analysis failed")
    
    def render_statistics(self):
        """Render system statistics"""
        st.markdown("### 📊 System Statistics")
        
        if st.button("Refresh Statistics", key="stats_button"):
            with st.spinner("Loading statistics..."):
                stats = self.make_api_request("/statistics")
            
            if stats:
                # Display fact statistics
                if "facts" in stats:
                    fact_stats = stats["facts"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Facts", fact_stats.get("total_facts", 0))
                    with col2:
                        st.metric("Facts with Embeddings", fact_stats.get("facts_with_embeddings", 0))
                    with col3:
                        st.metric("Unique Indicators", fact_stats.get("unique_indicators", 0))
                    with col4:
                        st.metric("Unique Districts", fact_stats.get("unique_districts", 0))
                
                # Display document statistics
                if "documents" in stats:
                    doc_stats = stats["documents"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Documents", doc_stats.get("total_documents", 0))
                    with col2:
                        st.metric("Source Types", doc_stats.get("unique_source_types", 0))
                
                # Display system status
                if "system_status" in stats:
                    st.success(f"System Status: {stats['system_status']}")
                
                if "last_updated" in stats:
                    st.info(f"Last Updated: {stats['last_updated']}")
            else:
                st.error("Failed to load statistics")
    
    def render_health_check(self):
        """Render health check section"""
        st.markdown("### 🏥 System Health")
        
        if st.button("Check Health", key="health_button"):
            with st.spinner("Checking system health..."):
                health = self.make_api_request("/health")
            
            if health:
                # Display health status
                status = health.get("status", "unknown")
                if status == "healthy":
                    st.success("✅ System is healthy")
                else:
                    st.error("❌ System is unhealthy")
                
                # Display component status
                col1, col2, col3 = st.columns(3)
                with col1:
                    pg_status = health.get("postgresql", "unknown")
                    if pg_status == "connected":
                        st.success("PostgreSQL: Connected")
                    else:
                        st.error("PostgreSQL: Disconnected")
                
                with col2:
                    neo4j_status = health.get("neo4j", "unknown")
                    if neo4j_status == "connected":
                        st.success("Neo4j: Connected")
                    else:
                        st.warning("Neo4j: Not Available")
                
                with col3:
                    embedding_status = health.get("embedding_model", "unknown")
                    if embedding_status == "available":
                        st.success("Embeddings: Available")
                    else:
                        st.warning("Embeddings: Not Available")
                
                # Display timestamp
                if "timestamp" in health:
                    st.info(f"Last Check: {health['timestamp']}")
            else:
                st.error("Health check failed")
    
    def render_main_content(self):
        """Render main dashboard content"""
        # Render header
        self.render_header()
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Search", "📈 Trends", "🏆 Comparison", "📊 Statistics", "🏥 Health"
        ])
        
        with tab1:
            self.render_search_results(self.render_sidebar())
        
        with tab2:
            self.render_trend_analysis()
        
        with tab3:
            self.render_comparison_analysis()
        
        with tab4:
            self.render_statistics()
        
        with tab5:
            self.render_health_check()
    
    def run(self):
        """Run the dashboard"""
        try:
            self.render_main_content()
        except Exception as e:
            st.error(f"Dashboard error: {e}")
            logger.error(f"Dashboard error: {e}")

def main():
    """Main function to run the dashboard"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize and run dashboard
    dashboard = APEducationDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
