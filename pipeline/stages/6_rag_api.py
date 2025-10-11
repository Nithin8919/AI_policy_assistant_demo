#!/usr/bin/env python3
"""
Stage 6: RAG API Server
FastAPI server for hybrid retrieval combining graph and vector search
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# FastAPI
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Vector embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    limit: int = 10
    include_graph: bool = True
    include_vector: bool = True
    filters: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    query_time: float
    search_method: str
    metadata: Dict[str, Any]

class FactResponse(BaseModel):
    fact_id: str
    indicator: str
    district: str
    year: str
    value: float
    unit: str
    source: str
    confidence: float
    span_text: str
    similarity_score: Optional[float] = None

class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    query: str
    execution_time: float

class RAGAPIServer:
    """Production-ready RAG API server for AP education policy intelligence"""
    
    def __init__(self):
        # Database configurations
        self.pg_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'ap_education_policy',
            'user': 'postgres',
            'password': 'password'
        }
        
        self.neo4j_config = {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password'
        }
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedding_model = None
            logger.warning("SentenceTransformers not available")
        
        # Initialize Neo4j driver
        self.neo4j_driver = None
        if NEO4J_AVAILABLE:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    self.neo4j_config['uri'],
                    auth=(self.neo4j_config['user'], self.neo4j_config['password'])
                )
            except Exception as e:
                logger.error(f"Neo4j connection failed: {e}")
        
        # Query templates
        self.query_templates = self._build_query_templates()
    
    def _build_query_templates(self) -> Dict[str, str]:
        """Build query templates for different search types"""
        return {
            'semantic_search': """
                SELECT 
                    f.fact_id,
                    f.indicator,
                    f.district,
                    f.year,
                    f.value,
                    f.unit,
                    f.source,
                    f.span_text,
                    f.confidence,
                    1 - (f.embedding <=> %s) as similarity_score
                FROM facts f
                WHERE f.embedding IS NOT NULL
                ORDER BY f.embedding <=> %s
                LIMIT %s
            """,
            
            'hybrid_search': """
                SELECT 
                    f.fact_id,
                    f.indicator,
                    f.district,
                    f.year,
                    f.value,
                    f.unit,
                    f.source,
                    f.span_text,
                    f.confidence,
                    (1 - (f.embedding <=> %s)) * 0.7 + 
                    CASE 
                        WHEN f.indicator ILIKE %s THEN 0.3
                        WHEN f.district ILIKE %s THEN 0.2
                        WHEN f.year ILIKE %s THEN 0.1
                        ELSE 0
                    END as combined_score
                FROM facts f
                WHERE f.embedding IS NOT NULL
                ORDER BY combined_score DESC
                LIMIT %s
            """,
            
            'graph_search': """
                MATCH (f:Fact)-[:MEASURED_BY]->(i:Indicator)
                MATCH (f)-[:LOCATED_IN]->(d:District)
                MATCH (f)-[:OBSERVED_IN]->(y:Year)
                MATCH (f)-[:REPORTED_BY]->(s:Source)
                WHERE i.name CONTAINS $indicator OR d.name CONTAINS $district
                RETURN f.fact_id as fact_id,
                       i.name as indicator,
                       d.name as district,
                       y.value as year,
                       f.value as value,
                       f.unit as unit,
                       s.name as source,
                       f.confidence as confidence
                ORDER BY f.confidence DESC
                LIMIT $limit
            """,
            
            'trend_analysis': """
                MATCH (f:Fact)-[:MEASURED_BY]->(i:Indicator {name: $indicator})
                MATCH (f)-[:LOCATED_IN]->(d:District)
                MATCH (f)-[:OBSERVED_IN]->(y:Year)
                RETURN d.name as district,
                       y.value as year,
                       f.value as value,
                       f.unit as unit
                ORDER BY d.name, y.value
            """,
            
            'comparison_analysis': """
                MATCH (f:Fact)-[:MEASURED_BY]->(i:Indicator)
                MATCH (f)-[:LOCATED_IN]->(d:District)
                MATCH (f)-[:OBSERVED_IN]->(y:Year {value: $year})
                WHERE i.name = $indicator
                RETURN d.name as district,
                       f.value as value,
                       f.unit as unit,
                       f.confidence as confidence
                ORDER BY f.value DESC
            """
        }
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using vector embeddings"""
        if not self.embedding_model:
            raise HTTPException(status_code=500, detail="Embedding model not available")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(self.query_templates['semantic_search'], 
                         (query_embedding, query_embedding, limit))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def hybrid_search(self, query: str, filters: Dict[str, Any] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword matching"""
        if not self.embedding_model:
            raise HTTPException(status_code=500, detail="Embedding model not available")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Extract filters
            indicator_filter = f"%{filters.get('indicator', '')}%" if filters else ""
            district_filter = f"%{filters.get('district', '')}%" if filters else ""
            year_filter = f"%{filters.get('year', '')}%" if filters else ""
            
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(self.query_templates['hybrid_search'], (
                query_embedding, indicator_filter, district_filter, year_filter, limit
            ))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def graph_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform graph-based search using Neo4j"""
        if not self.neo4j_driver:
            raise HTTPException(status_code=500, detail="Neo4j not available")
        
        try:
            with self.neo4j_driver.session() as session:
                # Extract keywords from query
                keywords = query.lower().split()
                indicator_keywords = [kw for kw in keywords if kw in ['ger', 'ner', 'gpi', 'ptr', 'dropout', 'retention']]
                district_keywords = [kw for kw in keywords if kw in ['visakhapatnam', 'vijayawada', 'guntur', 'nellore']]
                
                # Build graph query
                if indicator_keywords:
                    graph_query = self.query_templates['graph_search']
                    result = session.run(graph_query, 
                                        indicator=indicator_keywords[0],
                                        district=district_keywords[0] if district_keywords else "",
                                        limit=limit)
                else:
                    # Fallback to general search
                    graph_query = """
                        MATCH (f:Fact)-[:MEASURED_BY]->(i:Indicator)
                        MATCH (f)-[:LOCATED_IN]->(d:District)
                        MATCH (f)-[:OBSERVED_IN]->(y:Year)
                        MATCH (f)-[:REPORTED_BY]->(s:Source)
                        RETURN f.fact_id as fact_id,
                               i.name as indicator,
                               d.name as district,
                               y.value as year,
                               f.value as value,
                               f.unit as unit,
                               s.name as source,
                               f.confidence as confidence
                        ORDER BY f.confidence DESC
                        LIMIT $limit
                    """
                    result = session.run(graph_query, limit=limit)
                
                return [dict(record) for record in result]
        
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def trend_analysis(self, indicator: str) -> List[Dict[str, Any]]:
        """Analyze trends for a specific indicator"""
        if not self.neo4j_driver:
            raise HTTPException(status_code=500, detail="Neo4j not available")
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(self.query_templates['trend_analysis'], indicator=indicator)
                return [dict(record) for record in result]
        
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def comparison_analysis(self, indicator: str, year: str) -> List[Dict[str, Any]]:
        """Compare districts for a specific indicator and year"""
        if not self.neo4j_driver:
            raise HTTPException(status_code=500, detail="Neo4j not available")
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(self.query_templates['comparison_analysis'], 
                                  indicator=indicator, year=year)
                return [dict(record) for record in result]
        
        except Exception as e:
            logger.error(f"Comparison analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_fact_details(self, fact_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific fact"""
        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT f.*, d.filename, d.source_type
                FROM facts f
                LEFT JOIN documents d ON f.pdf_name = d.filename
                WHERE f.fact_id = %s
            """, (fact_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not result:
                raise HTTPException(status_code=404, detail="Fact not found")
            
            return dict(result)
        
        except Exception as e:
            logger.error(f"Fact details retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_similar_facts(self, fact_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find facts similar to a given fact"""
        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get reference fact embedding
            cursor.execute("SELECT embedding FROM facts WHERE fact_id = %s", (fact_id,))
            result = cursor.fetchone()
            
            if not result or not result['embedding']:
                raise HTTPException(status_code=404, detail="Fact embedding not found")
            
            reference_embedding = result['embedding']
            
            # Find similar facts
            cursor.execute("""
                SELECT 
                    f.fact_id,
                    f.indicator,
                    f.district,
                    f.year,
                    f.value,
                    f.unit,
                    f.source,
                    f.span_text,
                    f.confidence,
                    1 - (f.embedding <=> %s) as similarity_score
                FROM facts f
                WHERE f.fact_id != %s AND f.embedding IS NOT NULL
                ORDER BY f.embedding <=> %s
                LIMIT %s
            """, (reference_embedding, fact_id, reference_embedding, limit))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Similar facts search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            conn = psycopg2.connect(**self.pg_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get fact statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_facts,
                    COUNT(embedding) as facts_with_embeddings,
                    COUNT(DISTINCT indicator) as unique_indicators,
                    COUNT(DISTINCT district) as unique_districts,
                    COUNT(DISTINCT year) as unique_years,
                    COUNT(DISTINCT source) as unique_sources
                FROM facts
            """)
            
            fact_stats = cursor.fetchone()
            
            # Get document statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(DISTINCT source_type) as unique_source_types
                FROM documents
            """)
            
            doc_stats = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return {
                'facts': dict(fact_stats),
                'documents': dict(doc_stats),
                'system_status': 'operational',
                'last_updated': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
app = FastAPI(
    title="AP Education Policy Intelligence API",
    description="RAG API for Andhra Pradesh education policy data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG server
rag_server = RAGAPIServer()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AP Education Policy Intelligence API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/search", response_model=QueryResponse)
async def search(request: QueryRequest):
    """Hybrid search endpoint"""
    start_time = datetime.now()
    
    try:
        if request.include_vector and request.include_graph:
            # Hybrid search
            results = rag_server.hybrid_search(
                request.query, 
                request.filters, 
                request.limit
            )
            search_method = "hybrid"
        elif request.include_vector:
            # Vector search only
            results = rag_server.semantic_search(request.query, request.limit)
            search_method = "semantic"
        elif request.include_graph:
            # Graph search only
            results = rag_server.graph_search(request.query, request.limit)
            search_method = "graph"
        else:
            raise HTTPException(status_code=400, detail="At least one search method must be enabled")
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            results=results,
            total_count=len(results),
            query_time=query_time,
            search_method=search_method,
            metadata={
                "query": request.query,
                "filters": request.filters,
                "limit": request.limit
            }
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/facts/{fact_id}", response_model=FactResponse)
async def get_fact(fact_id: str):
    """Get fact details"""
    try:
        fact = rag_server.get_fact_details(fact_id)
        return FactResponse(**fact)
    except Exception as e:
        logger.error(f"Fact retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/facts/{fact_id}/similar")
async def get_similar_facts(fact_id: str, limit: int = Query(5, ge=1, le=20)):
    """Get similar facts"""
    try:
        results = rag_server.get_similar_facts(fact_id, limit)
        return {"similar_facts": results}
    except Exception as e:
        logger.error(f"Similar facts retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trends/{indicator}")
async def get_trends(indicator: str):
    """Get trend analysis for an indicator"""
    try:
        results = rag_server.trend_analysis(indicator)
        return {"trends": results}
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/compare/{indicator}/{year}")
async def compare_districts(indicator: str, year: str):
    """Compare districts for an indicator and year"""
    try:
        results = rag_server.comparison_analysis(indicator, year)
        return {"comparison": results}
    except Exception as e:
        logger.error(f"Comparison analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get system statistics"""
    try:
        stats = rag_server.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connections
        pg_status = "connected"
        try:
            conn = psycopg2.connect(**rag_server.pg_config)
            conn.close()
        except:
            pg_status = "disconnected"
        
        neo4j_status = "connected"
        if rag_server.neo4j_driver:
            try:
                with rag_server.neo4j_driver.session() as session:
                    session.run("RETURN 1")
            except:
                neo4j_status = "disconnected"
        else:
            neo4j_status = "not_available"
        
        return {
            "status": "healthy" if pg_status == "connected" else "unhealthy",
            "postgresql": pg_status,
            "neo4j": neo4j_status,
            "embedding_model": "available" if rag_server.embedding_model else "not_available",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run server
    uvicorn.run(
        "6_rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
