"""
FastAPI backend server for AI Policy Co-Pilot MVP
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import logging
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.retriever import WeaviateRetriever
from backend.embeddings import EmbeddingService
from backend.graph_manager import GraphManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AP Education Policy Co-Pilot API",
    description="RAG-based policy intelligence system with Weaviate",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services (in production, use dependency injection)
retriever = None
embedding_service = None
graph_manager = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    limit: int = 10
    include_graph: bool = True
    include_vector: bool = True
    filters: Optional[Dict[str, Any]] = None
    alpha: float = 0.7  # NEW: hybrid search balance

class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    graph_context: Optional[List[Dict[str, Any]]] = None
    processing_time: float

class DocumentRequest(BaseModel):
    document_type: str
    source_url: Optional[str] = None
    file_path: Optional[str] = None

class DocumentResponse(BaseModel):
    document_id: str
    status: str
    message: str

# Dependency functions
def get_retriever():
    global retriever
    if retriever is None:
        retriever = WeaviateRetriever()
    return retriever

def get_embedding_service():
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service

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
        "name": "AP Education Policy Co-Pilot API",
        "version": "2.0.0",
        "database": "Weaviate + Neo4j",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Weaviate
        weaviate_retriever = get_retriever()
        stats = weaviate_retriever.get_statistics()
        
        return {
            "status": "healthy",
            "database": "Weaviate",
            "facts_count": stats.get("total_facts", 0),
            "documents_count": stats.get("total_documents", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_policies(request: QueryRequest):
    """Search endpoint with hybrid search"""
    try:
        import time
        start_time = time.time()
        
        # Weaviate hybrid search
        weaviate_retriever = get_retriever()
        results = weaviate_retriever.search(
            query=request.query,
            limit=request.limit,
            filters=request.filters,
            alpha=request.alpha
        )
        
        # Get graph context if requested
        graph_context = None
        if request.include_graph and results:
            graph_mgr = get_graph_manager()
            entity_ids = [r['fact_id'] for r in results[:5]]
            graph_context = graph_mgr.get_entity_context(entity_ids)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            results=results,
            graph_context=graph_context,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Document management endpoints - TO BE IMPLEMENTED
# These would integrate with the Weaviate pipeline
@app.post("/documents", response_model=DocumentResponse)
async def add_document(request: DocumentRequest):
    """Add a new document to the knowledge base"""
    # This would trigger the data pipeline to process new documents
    raise HTTPException(status_code=501, detail="Document upload not implemented yet - use pipeline stages")

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        weaviate_retriever = get_retriever()
        graph_mgr = get_graph_manager()
        
        weaviate_stats = weaviate_retriever.get_statistics()
        graph_stats = graph_mgr.get_statistics()
        
        return {
            "weaviate": weaviate_stats,
            "knowledge_graph": graph_stats,
            "total_facts": weaviate_stats.get("total_facts", 0),
            "total_documents": weaviate_stats.get("total_documents", 0)
        }
    
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all documents in the knowledge base"""
    try:
        weaviate_retriever = get_retriever()
        client = weaviate_retriever.client
        doc_collection = client.collections.get("Document")
        
        response = doc_collection.query.fetch_objects(limit=100)
        documents = []
        for obj in response.objects:
            documents.append({
                'doc_id': obj.properties.get('doc_id'),
                'filename': obj.properties.get('filename'),
                'source_type': obj.properties.get('source_type'),
                'year': obj.properties.get('year'),
                'total_pages': obj.properties.get('total_pages')
            })
        
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document listing failed: {str(e)}")

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get details of a specific document"""
    try:
        weaviate_retriever = get_retriever()
        client = weaviate_retriever.client
        doc_collection = client.collections.get("Document")
        
        from weaviate.classes.query import Filter
        response = doc_collection.query.fetch_objects(
            filters=Filter.by_property("doc_id").equal(document_id),
            limit=1
        )
        
        if not response.objects:
            raise HTTPException(status_code=404, detail="Document not found")
        
        obj = response.objects[0]
        return {
            'doc_id': obj.properties.get('doc_id'),
            'filename': obj.properties.get('filename'),
            'source_type': obj.properties.get('source_type'),
            'year': obj.properties.get('year'),
            'total_pages': obj.properties.get('total_pages'),
            'extraction_method': obj.properties.get('extraction_method'),
            'file_path': obj.properties.get('file_path'),
            'created_at': obj.properties.get('created_at')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global retriever, graph_manager
    
    logger.info("Initializing backend services...")
    
    retriever = WeaviateRetriever()
    graph_manager = GraphManager()
    
    logger.info("Backend services initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global retriever, graph_manager
    
    if retriever:
        retriever.close()
    if graph_manager:
        graph_manager.close()
    
    logger.info("Backend services closed")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
