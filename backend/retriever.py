"""
Policy Retriever for RAG system
"""
import psycopg2
import psycopg2.extras
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class PolicyRetriever:
    """Retrieves relevant policy documents using vector similarity"""
    
    def __init__(self, db_config: Optional[Dict[str, str]] = None):
        self.db_config = db_config or {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'policy'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '1234')
        }
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test database connection
        self._test_connection()
    
    def _test_connection(self):
        """Test database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def retrieve(
        self,
        query_embedding: List[float],
        max_results: int = 5,
        include_vector: bool = True,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant policy documents using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            max_results: Maximum number of results to return
            include_vector: Whether to include vector similarity scores
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Query for similar spans using cosine similarity
            query = """
            SELECT 
                bt.bridge_id,
                bt.doc_id,
                bt.span_text,
                bt.span_start,
                bt.span_end,
                bt.entity_id,
                bt.relation_id,
                bt.confidence,
                bt.source_url,
                bt.created_at,
                1 - (bt.embedding <=> %s::vector) as similarity_score
            FROM bridge_table bt
            WHERE 1 - (bt.embedding <=> %s::vector) > %s
            ORDER BY bt.embedding <=> %s::vector
            LIMIT %s;
            """
            
            cursor.execute(query, (embedding_str, embedding_str, similarity_threshold, embedding_str, max_results))
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            retrieved_docs = []
            for row in results:
                doc = dict(row)
                if not include_vector:
                    doc.pop('similarity_score', None)
                retrieved_docs.append(doc)
            
            conn.close()
            
            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    def retrieve_by_entity(
        self,
        entity_id: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents by specific entity ID
        
        Args:
            entity_id: Entity identifier
            max_results: Maximum number of results
            
        Returns:
            List of documents containing the entity
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
            SELECT 
                bt.bridge_id,
                bt.doc_id,
                bt.span_text,
                bt.span_start,
                bt.span_end,
                bt.entity_id,
                bt.relation_id,
                bt.confidence,
                bt.source_url,
                bt.created_at
            FROM bridge_table bt
            WHERE bt.entity_id = %s
            ORDER BY bt.confidence DESC
            LIMIT %s;
            """
            
            cursor.execute(query, (entity_id, max_results))
            results = cursor.fetchall()
            
            retrieved_docs = [dict(row) for row in results]
            
            conn.close()
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for entity {entity_id}")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Entity retrieval failed: {e}")
            raise
    
    def retrieve_by_document(
        self,
        doc_id: str,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all spans for a specific document
        
        Args:
            doc_id: Document identifier
            max_results: Maximum number of results
            
        Returns:
            List of spans from the document
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
            SELECT 
                bt.bridge_id,
                bt.doc_id,
                bt.span_text,
                bt.span_start,
                bt.span_end,
                bt.entity_id,
                bt.relation_id,
                bt.confidence,
                bt.source_url,
                bt.created_at
            FROM bridge_table bt
            WHERE bt.doc_id = %s
            ORDER BY bt.span_start
            LIMIT %s;
            """
            
            cursor.execute(query, (doc_id, max_results))
            results = cursor.fetchall()
            
            retrieved_docs = [dict(row) for row in results]
            
            conn.close()
            
            logger.info(f"Retrieved {len(retrieved_docs)} spans for document {doc_id}")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            raise
    
    def hybrid_retrieve(
        self,
        query_text: str,
        query_embedding: List[float],
        max_results: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining vector similarity and keyword matching
        
        Args:
            query_text: Original query text for keyword matching
            query_embedding: Query embedding for vector similarity
            max_results: Maximum number of results
            vector_weight: Weight for vector similarity score
            keyword_weight: Weight for keyword matching score
            
        Returns:
            List of documents with combined relevance scores
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Extract keywords from query
            keywords = query_text.lower().split()
            keyword_pattern = '|'.join(keywords)
            
            # Hybrid query combining vector similarity and keyword matching
            query = """
            SELECT 
                bt.bridge_id,
                bt.doc_id,
                bt.span_text,
                bt.span_start,
                bt.span_end,
                bt.entity_id,
                bt.relation_id,
                bt.confidence,
                bt.source_url,
                bt.created_at,
                (%s * (1 - (bt.embedding <=> %s::vector))) + 
                (%s * CASE 
                    WHEN bt.span_text ILIKE ANY(ARRAY[%s])
                    THEN 1.0 
                    ELSE 0.0 
                END) as hybrid_score
            FROM bridge_table bt
            WHERE (1 - (bt.embedding <=> %s::vector)) > 0.5 
               OR bt.span_text ILIKE ANY(ARRAY[%s])
            ORDER BY hybrid_score DESC
            LIMIT %s;
            """
            
            # Create keyword patterns for ILIKE matching
            keyword_patterns = [f'%{kw}%' for kw in keywords]
            
            cursor.execute(query, (
                vector_weight, embedding_str,
                keyword_weight, keyword_patterns,
                embedding_str, keyword_patterns,
                max_results
            ))
            
            results = cursor.fetchall()
            retrieved_docs = [dict(row) for row in results]
            
            conn.close()
            
            logger.info(f"Hybrid retrieval returned {len(retrieved_docs)} documents")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            raise
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get total spans count
            cursor.execute("SELECT COUNT(*) FROM bridge_table")
            total_spans = cursor.fetchone()[0]
            
            # Get unique documents count
            cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM bridge_table")
            unique_docs = cursor.fetchone()[0]
            
            # Get unique entities count
            cursor.execute("SELECT COUNT(DISTINCT entity_id) FROM bridge_table WHERE entity_id IS NOT NULL")
            unique_entities = cursor.fetchone()[0]
            
            # Get average confidence
            cursor.execute("SELECT AVG(confidence) FROM bridge_table")
            avg_confidence = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'total_spans': total_spans,
                'unique_documents': unique_docs,
                'unique_entities': unique_entities,
                'average_confidence': round(avg_confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"Failed to get retrieval stats: {e}")
            return {}
