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

from backend.retriever import PolicyRetriever
from backend.embeddings import EmbeddingService
from backend.graph_manager import GraphManager
from backend.bridge_table import BridgeTableManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Policy Co-Pilot API",
    description="REST API for Andhra Pradesh Education Policy Intelligence",
    version="1.0.0"
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
bridge_manager = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    max_results: int = 5
    include_graph: bool = True
    include_vector: bool = True

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
        retriever = PolicyRetriever()
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

def get_bridge_manager():
    global bridge_manager
    if bridge_manager is None:
        bridge_manager = BridgeTableManager()
    return bridge_manager

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Policy Co-Pilot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "documents": "/documents",
            "health": "/health",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connections
        bridge_mgr = get_bridge_manager()
        graph_mgr = get_graph_manager()
        
        bridge_status = bridge_mgr.check_connection()
        graph_status = graph_mgr.check_connection()
        
        return {
            "status": "healthy",
            "services": {
                "bridge_table": bridge_status,
                "knowledge_graph": graph_status,
                "embedding_service": "healthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_policies(
    request: QueryRequest,
    retriever: PolicyRetriever = Depends(get_retriever),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    graph_manager: GraphManager = Depends(get_graph_manager)
):
    """Query the policy knowledge base"""
    import time
    start_time = time.time()
    
    try:
        # Generate embedding for query
        query_embedding = embedding_service.encode(request.query)
        
        # Retrieve relevant documents
        results = retriever.retrieve(
            query_embedding=query_embedding,
            max_results=request.max_results,
            include_vector=request.include_vector
        )
        
        # Get graph context if requested
        graph_context = None
        if request.include_graph and results:
            entity_ids = [r.get('entity_id') for r in results if r.get('entity_id')]
            if entity_ids:
                graph_context = graph_manager.get_entity_context(entity_ids)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            results=results,
            graph_context=graph_context,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/documents", response_model=DocumentResponse)
async def add_document(
    request: DocumentRequest,
    bridge_manager: BridgeTableManager = Depends(get_bridge_manager)
):
    """Add a new document to the knowledge base"""
    try:
        # This would integrate with the data pipeline
        document_id = bridge_manager.add_document(
            document_type=request.document_type,
            source_url=request.source_url,
            file_path=request.file_path
        )
        
        return DocumentResponse(
            document_id=document_id,
            status="success",
            message="Document added successfully"
        )
        
    except Exception as e:
        logger.error(f"Document addition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.get("/stats")
async def get_statistics(
    bridge_manager: BridgeTableManager = Depends(get_bridge_manager),
    graph_manager: GraphManager = Depends(get_graph_manager)
):
    """Get system statistics"""
    try:
        bridge_stats = bridge_manager.get_statistics()
        graph_stats = graph_manager.get_statistics()
        
        return {
            "bridge_table": bridge_stats,
            "knowledge_graph": graph_stats,
            "total_documents": bridge_stats.get("total_documents", 0),
            "total_entities": bridge_stats.get("total_entities", 0),
            "total_relations": bridge_stats.get("total_relations", 0)
        }
        
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")

@app.get("/documents")
async def list_documents(
    bridge_manager: BridgeTableManager = Depends(get_bridge_manager)
):
    """List all documents in the knowledge base"""
    try:
        documents = bridge_manager.list_documents()
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document listing failed: {str(e)}")

@app.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    bridge_manager: BridgeTableManager = Depends(get_bridge_manager)
):
    """Get details of a specific document"""
    try:
        document = bridge_manager.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting AI Policy Co-Pilot API...")
    
    try:
        # Initialize services
        global retriever, embedding_service, graph_manager, bridge_manager
        
        embedding_service = EmbeddingService()
        graph_manager = GraphManager()
        bridge_manager = BridgeTableManager()
        retriever = PolicyRetriever()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Policy Co-Pilot API...")
    
    try:
        if graph_manager:
            graph_manager.close()
        if bridge_manager:
            bridge_manager.close()
        
        logger.info("Shutdown completed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
