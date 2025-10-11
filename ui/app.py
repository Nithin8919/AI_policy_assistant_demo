"""
Streamlit UI for AI Policy Co-Pilot MVP
"""
import streamlit as st
import psycopg2
import psycopg2.extras
import json
import time
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.retriever import PolicyRetriever
from backend.embeddings import EmbeddingService
from backend.graph_manager import GraphManager
from backend.bridge_table import BridgeTableManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Policy Co-Pilot - Andhra Pradesh Education",
    page_icon="🎓",
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
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .entity-tag {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin: 0.1rem;
        display: inline-block;
    }
    .relation-tag {
        background-color: #f3e5f5;
        color: #7b1fa2;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin: 0.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize services
@st.cache_resource
def initialize_services():
    """Initialize backend services"""
    try:
        retriever = PolicyRetriever()
        embedding_service = EmbeddingService()
        graph_manager = GraphManager()
        bridge_manager = BridgeTableManager()
        
        return retriever, embedding_service, graph_manager, bridge_manager
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return None, None, None, None

# Main header
st.markdown('<div class="main-header">🎓 AI Policy Co-Pilot</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Andhra Pradesh School Education Policy Intelligence</div>', unsafe_allow_html=True)

# Initialize services
retriever, embedding_service, graph_manager, bridge_manager = initialize_services()

if retriever is None:
    st.error("❌ Failed to initialize backend services. Please check your database connections.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## 🔧 Configuration")
    
    # Query parameters
    max_results = st.slider("Maximum Results", 1, 20, 5)
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7)
    include_graph = st.checkbox("Include Graph Context", value=True)
    include_vector = st.checkbox("Include Vector Search", value=True)
    
    st.markdown("---")
    
    # System status
    st.markdown("## 📊 System Status")
    
    try:
        # Check database connections
        bridge_status = bridge_manager.check_connection()
        graph_status = graph_manager.check_connection()
        
        st.markdown(f"**Bridge Table:** {'✅ Connected' if bridge_status else '❌ Disconnected'}")
        st.markdown(f"**Knowledge Graph:** {'✅ Connected' if graph_status else '❌ Disconnected'}")
        
        # Get statistics
        bridge_stats = bridge_manager.get_statistics()
        graph_stats = graph_manager.get_statistics()
        
        st.markdown(f"**Documents:** {bridge_stats.get('total_documents', 0)}")
        st.markdown(f"**Entities:** {bridge_stats.get('total_entities', 0)}")
        st.markdown(f"**Relations:** {bridge_stats.get('total_relations', 0)}")
        
    except Exception as e:
        st.error(f"Status check failed: {e}")

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Query", "📊 Analytics", "📚 Documents", "🕸️ Knowledge Graph", "⚙️ Settings"])

# Query Tab
with tab1:
    st.markdown('<div class="sub-header">Policy Query Interface</div>', unsafe_allow_html=True)
    
    # Query input
    query = st.text_area(
        "Ask a question about Andhra Pradesh education policies:",
        placeholder="e.g., What are the guidelines for teacher recruitment in government schools?",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    with col3:
        export_button = st.button("📥 Export Results", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if search_button and query:
        with st.spinner("Searching policy database..."):
            try:
                start_time = time.time()
                
                # Generate embedding for query
                query_embedding = embedding_service.encode(query)
                
                # Retrieve relevant documents
                results = retriever.retrieve(
                    query_embedding=query_embedding,
                    max_results=max_results,
                    include_vector=include_vector
                )
                
                # Get graph context if requested
                graph_context = None
                if include_graph and results:
                    entity_ids = [r.get('entity_id') for r in results if r.get('entity_id')]
                    if entity_ids:
                        graph_context = graph_manager.get_entity_context(entity_ids)
                
                processing_time = time.time() - start_time
                
                # Display results
                st.markdown(f"**Found {len(results)} relevant results in {processing_time:.2f} seconds**")
                
                if results:
                    for i, result in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="result-card">
                                <h4>Result {i}</h4>
                                <p><strong>Document:</strong> {result.get('doc_id', 'Unknown')}</p>
                                <p><strong>Text:</strong> {result.get('span_text', 'No text available')}</p>
                                <p><strong>Confidence:</strong> {result.get('confidence', 0):.3f}</p>
                                <p><strong>Source:</strong> {result.get('source_url', 'No source')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Display graph context
                    if graph_context:
                        st.markdown('<div class="sub-header">🕸️ Related Knowledge Graph Context</div>', unsafe_allow_html=True)
                        
                        for context in graph_context[:3]:  # Show first 3 contexts
                            entity = context.get('entity', {})
                            connected_nodes = context.get('connected_nodes', [])
                            
                            st.markdown(f"**Entity:** {entity.get('name', 'Unknown')}")
                            
                            if connected_nodes:
                                st.markdown("**Connected Entities:**")
                                for node in connected_nodes[:5]:  # Show first 5 connected nodes
                                    st.markdown(f"- {node.get('name', 'Unknown')} ({node.get('labels', ['Unknown'])[0]})")
                    
                    # Export results
                    if export_button:
                        export_data = {
                            'query': query,
                            'results': results,
                            'graph_context': graph_context,
                            'processing_time': processing_time,
                            'parameters': {
                                'max_results': max_results,
                                'similarity_threshold': similarity_threshold,
                                'include_graph': include_graph,
                                'include_vector': include_vector
                            }
                        }
                        
                        st.download_button(
                            label="📥 Download Results as JSON",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"policy_query_results_{int(time.time())}.json",
                            mime="application/json"
                        )
                
                else:
                    st.warning("No relevant results found. Try rephrasing your query or adjusting the similarity threshold.")
                    
            except Exception as e:
                st.error(f"Query failed: {e}")
                logger.error(f"Query failed: {e}")

# Analytics Tab
with tab2:
    st.markdown('<div class="sub-header">Policy Analytics Dashboard</div>', unsafe_allow_html=True)
    
    try:
        # Get statistics
        bridge_stats = bridge_manager.get_statistics()
        graph_stats = graph_manager.get_statistics()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Documents", bridge_stats.get('total_documents', 0))
        
        with col2:
            st.metric("🏷️ Entities", bridge_stats.get('total_entities', 0))
        
        with col3:
            st.metric("🔗 Relations", bridge_stats.get('total_relations', 0))
        
        with col4:
            st.metric("📊 Avg Confidence", f"{bridge_stats.get('average_confidence', 0):.3f}")
        
        # Entity type distribution
        st.markdown('<div class="sub-header">Entity Type Distribution</div>', unsafe_allow_html=True)
        
        entity_types = bridge_stats.get('entity_type_distribution', {})
        if entity_types:
            df_entities = pd.DataFrame(list(entity_types.items()), columns=['Entity Type', 'Count'])
            fig_entities = px.pie(df_entities, values='Count', names='Entity Type', title="Entity Types")
            st.plotly_chart(fig_entities, use_container_width=True)
        else:
            st.info("No entity type data available")
        
        # Document type distribution
        st.markdown('<div class="sub-header">Document Type Distribution</div>', unsafe_allow_html=True)
        
        doc_types = bridge_stats.get('document_type_distribution', {})
        if doc_types:
            df_docs = pd.DataFrame(list(doc_types.items()), columns=['Document Type', 'Count'])
            fig_docs = px.bar(df_docs, x='Document Type', y='Count', title="Document Types")
            st.plotly_chart(fig_docs, use_container_width=True)
        else:
            st.info("No document type data available")
        
        # Graph statistics
        if graph_stats:
            st.markdown('<div class="sub-header">Knowledge Graph Statistics</div>', unsafe_allow_html=True)
            
            node_counts = graph_stats.get('node_counts', {})
            if node_counts:
                df_nodes = pd.DataFrame(list(node_counts.items()), columns=['Node Type', 'Count'])
                fig_nodes = px.bar(df_nodes, x='Node Type', y='Count', title="Knowledge Graph Nodes")
                st.plotly_chart(fig_nodes, use_container_width=True)
            
            st.metric("Total Graph Relations", graph_stats.get('total_relations', 0))
        
    except Exception as e:
        st.error(f"Analytics failed: {e}")

# Documents Tab
with tab3:
    st.markdown('<div class="sub-header">Document Management</div>', unsafe_allow_html=True)
    
    try:
        # List documents
        documents = bridge_manager.list_documents()
        
        if documents:
            st.markdown(f"**Total Documents: {len(documents)}**")
            
            # Document table
            df_docs = pd.DataFrame(documents)
            st.dataframe(df_docs, use_container_width=True)
            
            # Document details
            if len(documents) > 0:
                selected_doc = st.selectbox(
                    "Select a document to view details:",
                    options=[doc['doc_id'] for doc in documents]
                )
                
                if selected_doc:
                    doc_details = bridge_manager.get_document(selected_doc)
                    if doc_details:
                        st.markdown("### Document Details")
                        st.json(doc_details)
                        
                        # Get document spans
                        spans = retriever.retrieve_by_document(selected_doc, max_results=50)
                        if spans:
                            st.markdown(f"### Document Spans ({len(spans)} total)")
                            
                            for span in spans[:10]:  # Show first 10 spans
                                st.markdown(f"""
                                <div class="result-card">
                                    <p><strong>Text:</strong> {span.get('span_text', 'No text')}</p>
                                    <p><strong>Entity:</strong> {span.get('entity_id', 'None')}</p>
                                    <p><strong>Confidence:</strong> {span.get('confidence', 0):.3f}</p>
                                </div>
                                """, unsafe_allow_html=True)
        else:
            st.info("No documents found in the database")
    
    except Exception as e:
        st.error(f"Document management failed: {e}")

# Knowledge Graph Tab
with tab4:
    st.markdown('<div class="sub-header">Knowledge Graph Explorer</div>', unsafe_allow_html=True)
    
    try:
        # Graph statistics
        graph_stats = graph_manager.get_statistics()
        
        if graph_stats:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Nodes", graph_stats.get('total_nodes', 0))
            
            with col2:
                st.metric("Total Relations", graph_stats.get('total_relations', 0))
            
            # Node type distribution
            node_counts = graph_stats.get('node_counts', {})
            if node_counts:
                st.markdown("### Node Type Distribution")
                df_nodes = pd.DataFrame(list(node_counts.items()), columns=['Node Type', 'Count'])
                fig_nodes = px.pie(df_nodes, values='Count', names='Node Type', title="Knowledge Graph Nodes")
                st.plotly_chart(fig_nodes, use_container_width=True)
        
        # Graph queries
        st.markdown("### Graph Queries")
        
        query_type = st.selectbox(
            "Query Type:",
            ["Entity Context", "Related Entities", "Entity Paths"]
        )
        
        if query_type == "Entity Context":
            entity_id = st.text_input("Enter Entity ID:")
            if entity_id and st.button("Get Context"):
                context = graph_manager.get_entity_context([entity_id])
                if context:
                    st.json(context[0])
                else:
                    st.warning("No context found for this entity")
        
        elif query_type == "Related Entities":
            entity_id = st.text_input("Enter Entity ID:")
            max_relations = st.slider("Max Relations", 1, 20, 10)
            if entity_id and st.button("Get Related Entities"):
                related = graph_manager.find_related_entities(entity_id, max_results=max_relations)
                if related:
                    for rel in related:
                        st.markdown(f"- **{rel['entity_id']}** ({rel['entity_type']}) - {rel['relation_type']}")
                else:
                    st.warning("No related entities found")
        
        elif query_type == "Entity Paths":
            col1, col2 = st.columns(2)
            with col1:
                start_entity = st.text_input("Start Entity ID:")
            with col2:
                end_entity = st.text_input("End Entity ID:")
            
            if start_entity and end_entity and st.button("Find Paths"):
                paths = graph_manager.get_entity_paths(start_entity, end_entity)
                if paths:
                    for i, path in enumerate(paths):
                        st.markdown(f"**Path {i+1}** (Length: {path['path_length']})")
                        for node in path['nodes']:
                            st.markdown(f"- {node['name']} ({node['type']})")
                else:
                    st.warning("No paths found between these entities")
    
    except Exception as e:
        st.error(f"Knowledge graph exploration failed: {e}")

# Settings Tab
with tab5:
    st.markdown('<div class="sub-header">System Settings</div>', unsafe_allow_html=True)
    
    # Database configuration
    st.markdown("### Database Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**PostgreSQL Settings**")
        postgres_host = st.text_input("Host", value="localhost")
        postgres_port = st.text_input("Port", value="5432")
        postgres_db = st.text_input("Database", value="policy")
        postgres_user = st.text_input("User", value="postgres")
        postgres_password = st.text_input("Password", type="password", value="1234")
    
    with col2:
        st.markdown("**Neo4j Settings**")
        neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        neo4j_user = st.text_input("Neo4j User", value="neo4j")
        neo4j_password = st.text_input("Neo4j Password", type="password", value="password")
    
    # Test connections
    if st.button("Test Database Connections"):
        try:
            # Test PostgreSQL
            import psycopg2
            conn = psycopg2.connect(
                host=postgres_host,
                port=postgres_port,
                database=postgres_db,
                user=postgres_user,
                password=postgres_password
            )
            conn.close()
            st.success("✅ PostgreSQL connection successful")
        except Exception as e:
            st.error(f"❌ PostgreSQL connection failed: {e}")
        
        try:
            # Test Neo4j
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            st.success("✅ Neo4j connection successful")
        except Exception as e:
            st.error(f"❌ Neo4j connection failed: {e}")
    
    # System information
    st.markdown("### System Information")
    
    try:
        import platform
        import sys
        
        st.markdown(f"**Python Version:** {sys.version}")
        st.markdown(f"**Platform:** {platform.platform()}")
        st.markdown(f"**Streamlit Version:** {st.__version__}")
        
        # Backend service versions
        try:
            import sentence_transformers
            st.markdown(f"**Sentence Transformers:** {sentence_transformers.__version__}")
        except:
            st.markdown("**Sentence Transformers:** Not available")
        
        try:
            import psycopg2
            st.markdown(f"**psycopg2:** {psycopg2.__version__}")
        except:
            st.markdown("**psycopg2:** Not available")
        
        try:
            import neo4j
            st.markdown(f"**Neo4j Driver:** {neo4j.__version__}")
        except:
            st.markdown("**Neo4j Driver:** Not available")
    
    except Exception as e:
        st.error(f"Failed to get system information: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        AI Policy Co-Pilot MVP - Andhra Pradesh School Education Policy Intelligence<br>
        Built with Streamlit, PostgreSQL, Neo4j, and Sentence Transformers
    </div>
    """,
    unsafe_allow_html=True
)

