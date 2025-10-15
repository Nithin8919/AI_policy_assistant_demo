"""
FastAPI backend server with Gemini RAG and Citation Support
Enhanced version with Google Gemini API for intelligent responses
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import os
from pathlib import Path
import sys
import time

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.gemini_rag_service import GeminiRAGService, RAGResponse, Citation
from backend.retriever import WeaviateRetriever
from backend.graph_manager import GraphManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AP Education Policy Co-Pilot API with Gemini",
    description="RAG-based policy intelligence system with Gemini AI and citations",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
gemini_service = None
retriever = None
graph_manager = None

# Pydantic models
class GeminiQueryRequest(BaseModel):
    query: str
    limit: int = 10
    include_citations: bool = True
    filters: Optional[Dict[str, Any]] = None

class CitationModel(BaseModel):
    id: str
    source_document: str
    district: str
    indicator: str
    year: int
    page_number: Optional[int] = None
    confidence_score: float
    excerpt: str
    url: Optional[str] = None

class GeminiQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationModel]
    retrieval_stats: Dict[str, Any]
    confidence_score: float
    processing_time: float

class LegacyQueryRequest(BaseModel):
    query: str
    limit: int = 10
    include_graph: bool = True
    include_vector: bool = True
    filters: Optional[Dict[str, Any]] = None
    alpha: float = 0.7

class LegacyQueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    graph_context: Optional[List[Dict[str, Any]]] = None
    processing_time: float

# Dependency functions
def get_gemini_service():
    global gemini_service
    if gemini_service is None:
        api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyCVtumVmWMJDNExIcmKTUsy_iPLB-NDQ7A')
        if not api_key or api_key == 'your_gemini_api_key_here':
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        gemini_service = GeminiRAGService(api_key)
    return gemini_service

def get_retriever():
    global retriever
    if retriever is None:
        retriever = WeaviateRetriever()
    return retriever

def get_graph_manager():
    global graph_manager
    if graph_manager is None:
        graph_manager = GraphManager()
    return graph_manager

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AP Education Policy Co-Pilot API with Gemini",
        "version": "3.0.0",
        "features": ["Gemini AI", "Citations", "Weaviate", "Neo4j"],
        "database": "Weaviate + Neo4j",
        "llm": "Google Gemini",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Gemini service
        gemini_svc = get_gemini_service()
        health_status = gemini_svc.health_check()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": health_status,
            "version": "3.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.post("/query/gemini", response_model=GeminiQueryResponse)
async def query_with_gemini(
    request: GeminiQueryRequest,
    gemini_service: GeminiRAGService = Depends(get_gemini_service)
):
    """
    Query with Gemini AI for intelligent responses and citations
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing Gemini query: {request.query}")
        
        # Process query with Gemini
        rag_response = gemini_service.query(
            user_query=request.query,
            limit=request.limit,
            filters=request.filters,
            include_citations=request.include_citations
        )
        
        # Convert to response model
        citations = [
            CitationModel(
                id=cite.id,
                source_document=cite.source_document,
                district=cite.district,
                indicator=cite.indicator,
                year=cite.year,
                page_number=cite.page_number,
                confidence_score=cite.confidence_score,
                excerpt=cite.excerpt,
                url=cite.url
            )
            for cite in rag_response.citations
        ]
        
        processing_time = time.time() - start_time
        
        response = GeminiQueryResponse(
            query=rag_response.query,
            answer=rag_response.answer,
            citations=citations,
            retrieval_stats=rag_response.retrieval_stats,
            confidence_score=rag_response.confidence_score,
            processing_time=processing_time
        )
        
        logger.info(f"Completed Gemini query in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Gemini query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/query", response_model=LegacyQueryResponse)
async def legacy_query(
    request: LegacyQueryRequest,
    weaviate_retriever: WeaviateRetriever = Depends(get_retriever)
):
    """
    Legacy query endpoint for backward compatibility
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing legacy query: {request.query}")
        
        # Search Weaviate
        search_results = weaviate_retriever.search(
            query=request.query,
            limit=request.limit,
            filters=request.filters,
            alpha=request.alpha
        )
        
        # Get graph context if requested
        graph_context = None
        if request.include_graph:
            try:
                graph_mgr = get_graph_manager()
                graph_context = graph_mgr.get_related_entities(request.query, limit=5)
            except Exception as e:
                logger.warning(f"Graph context failed: {e}")
        
        processing_time = time.time() - start_time
        
        return LegacyQueryResponse(
            query=request.query,
            results=search_results,
            graph_context=graph_context,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Legacy query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/citations/{citation_id}")
async def get_citation_details(citation_id: str):
    """Get detailed information for a specific citation"""
    try:
        # This would typically fetch from a citation store
        # For now, return basic info
        return {
            "citation_id": citation_id,
            "status": "found",
            "details": {
                "message": "Citation details endpoint - implementation depends on citation storage"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Citation not found: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        retriever_svc = get_retriever()
        stats = retriever_svc.get_statistics()
        
        return {
            "weaviate_stats": stats,
            "api_version": "3.0.0",
            "features": ["Gemini AI", "Citations", "Hybrid Search"]
        }
    except Exception as e:
        logger.error(f"Stats endpoint failed: {e}")
        return {"error": str(e)}

@app.get("/export/citations")
async def export_citations():
    """Export citations in various formats"""
    return {
        "message": "Citation export feature - implementation depends on requirements",
        "supported_formats": ["JSON", "CSV", "BibTeX", "APA"]
    }

# Development endpoints
@app.post("/test/gemini")
async def test_gemini():
    """Test Gemini API connectivity"""
    try:
        gemini_svc = get_gemini_service()
        test_response = gemini_svc.query("What is the status of education in Andhra Pradesh?", limit=3)
        
        return {
            "status": "success",
            "test_query": "What is the status of education in Andhra Pradesh?",
            "response_length": len(test_response.answer),
            "citations_count": len(test_response.citations),
            "confidence": test_response.confidence_score
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    # Set environment variable for Gemini API
    if not os.getenv('GEMINI_API_KEY'):
        os.environ['GEMINI_API_KEY'] = 'AIzaSyCVtumVmWMJDNExIcmKTUsy_iPLB-NDQ7A'
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )