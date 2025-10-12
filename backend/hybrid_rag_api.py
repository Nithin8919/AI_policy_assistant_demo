#!/usr/bin/env python3
"""
Hybrid RAG API for AP Policy Co-Pilot
FastAPI backend with vector search + graph RAG + bridge tables
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'pipeline' / 'utils'))

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from hybrid_rag_system import HybridRAGSystem, HybridSearchQuery, SearchResult

logger = logging.getLogger(__name__)

# Pydantic models for API
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    district_filter: Optional[str] = Field(None, description="Filter by district")
    indicator_filter: Optional[str] = Field(None, description="Filter by indicator")
    year_filter: Optional[int] = Field(None, description="Filter by specific year")
    year_range: Optional[List[int]] = Field(None, description="Filter by year range [start, end]")
    search_depth: int = Field(2, description="Graph traversal depth", ge=1, le=5)
    include_graph_context: bool = Field(True, description="Include graph context expansion")
    include_bridge_expansion: bool = Field(True, description="Include bridge table expansion")
    max_results: int = Field(10, description="Maximum number of results", ge=1, le=50)

class SearchResponse(BaseModel):
    fact_id: str
    content: str
    district: str
    indicator: str
    year: int
    value: float
    unit: str
    confidence_score: float
    source: str
    graph_context: Optional[Dict[str, Any]] = None
    bridge_connections: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]

class StatsResponse(BaseModel):
    total_facts: int
    total_bridges: int
    districts_covered: int
    indicators_covered: int
    year_range: List[int]
    last_updated: str

# Initialize FastAPI app
app = FastAPI(
    title="AP Policy Co-Pilot - Hybrid RAG API",
    description="AI-powered policy intelligence with vector search, graph RAG, and bridge tables",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system: Optional[HybridRAGSystem] = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    
    logger.info("🚀 Starting Hybrid RAG API")
    
    try:
        # Configuration for hybrid RAG system
        config = {
            'weaviate_url': 'http://localhost:8080',
            'neo4j_uri': 'bolt://localhost:7687',
            'neo4j_user': 'neo4j',
            'neo4j_password': 'password',
            'embedding_model': 'all-MiniLM-L6-v2',
            'data_dir': 'data'
        }
        
        rag_system = HybridRAGSystem(config)
        logger.info("✅ Hybrid RAG system initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global rag_system
    
    if rag_system:
        rag_system.close()
        logger.info("🔒 RAG system connections closed")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global rag_system
    
    components = {}
    
    # Check Weaviate
    try:
        if rag_system and rag_system.weaviate_client.is_ready():
            components["weaviate"] = "healthy"
        else:
            components["weaviate"] = "unhealthy"
    except:
        components["weaviate"] = "unhealthy"
    
    # Check Neo4j
    try:
        if rag_system:
            with rag_system.neo4j_driver.session() as session:
                session.run("RETURN 1")
            components["neo4j"] = "healthy"
        else:
            components["neo4j"] = "unhealthy"
    except:
        components["neo4j"] = "unhealthy"
    
    # Check bridge tables
    try:
        if rag_system and rag_system.bridge_manager:
            components["bridge_tables"] = "healthy"
        else:
            components["bridge_tables"] = "unhealthy"
    except:
        components["bridge_tables"] = "unhealthy"
    
    overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        components=components
    )

@app.post("/search", response_model=List[SearchResponse])
async def hybrid_search(request: SearchRequest):
    """Hybrid search endpoint using vector + graph + bridge tables"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        logger.info(f"🔍 Hybrid search: '{request.query[:50]}...'")
        
        # Convert year_range to tuple if provided
        year_range = None
        if request.year_range and len(request.year_range) == 2:
            year_range = (request.year_range[0], request.year_range[1])
        
        # Create hybrid search query
        search_query = HybridSearchQuery(
            query_text=request.query,
            district_filter=request.district_filter,
            indicator_filter=request.indicator_filter,
            year_filter=request.year_filter,
            year_range=year_range,
            search_depth=request.search_depth,
            include_graph_context=request.include_graph_context,
            include_bridge_expansion=request.include_bridge_expansion,
            max_results=request.max_results
        )
        
        # Perform hybrid search
        results = rag_system.search(search_query)
        
        # Convert to response format
        response_results = []
        for result in results:
            response_results.append(SearchResponse(
                fact_id=result.fact_id,
                content=result.content,
                district=result.district,
                indicator=result.indicator,
                year=result.year,
                value=result.value,
                unit=result.unit,
                confidence_score=result.confidence_score,
                source=result.source,
                graph_context=result.graph_context,
                bridge_connections=result.bridge_connections,
                metadata=result.metadata
            ))
        
        logger.info(f"✅ Returned {len(response_results)} results")
        return response_results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/search", response_model=List[SearchResponse])
async def hybrid_search_get(
    q: str = Query(..., description="Search query"),
    district: Optional[str] = Query(None, description="Filter by district"),
    indicator: Optional[str] = Query(None, description="Filter by indicator"),
    year: Optional[int] = Query(None, description="Filter by year"),
    depth: int = Query(2, description="Search depth", ge=1, le=5),
    limit: int = Query(10, description="Max results", ge=1, le=50)
):
    """GET endpoint for simple searches"""
    request = SearchRequest(
        query=q,
        district_filter=district,
        indicator_filter=indicator,
        year_filter=year,
        search_depth=depth,
        max_results=limit
    )
    
    return await hybrid_search(request)

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Get stats from Weaviate
        facts_response = (
            rag_system.weaviate_client.query
            .aggregate("Fact")
            .with_meta_count()
            .do()
        )
        
        total_facts = facts_response.get("data", {}).get("Aggregate", {}).get("Fact", [{}])[0].get("meta", {}).get("count", 0)
        
        bridges_response = (
            rag_system.weaviate_client.query
            .aggregate("BridgeTable")
            .with_meta_count()
            .do()
        )
        
        total_bridges = bridges_response.get("data", {}).get("Aggregate", {}).get("BridgeTable", [{}])[0].get("meta", {}).get("count", 0)
        
        # Get unique districts and indicators
        districts_response = (
            rag_system.weaviate_client.query
            .aggregate("Fact")
            .with_group_by_filter(["district"])
            .do()
        )
        
        districts_count = len(districts_response.get("data", {}).get("Aggregate", {}).get("Fact", []))
        
        indicators_response = (
            rag_system.weaviate_client.query
            .aggregate("Fact") 
            .with_group_by_filter(["indicator"])
            .do()
        )
        
        indicators_count = len(indicators_response.get("data", {}).get("Aggregate", {}).get("Fact", []))
        
        # Get year range (simplified)
        year_range = [2020, 2023]  # Default range
        
        return StatsResponse(
            total_facts=total_facts,
            total_bridges=total_bridges,
            districts_covered=districts_count,
            indicators_covered=indicators_count,
            year_range=year_range,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats query failed: {str(e)}")

@app.get("/districts")
async def get_districts():
    """Get list of available districts"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        response = (
            rag_system.weaviate_client.query
            .aggregate("Fact")
            .with_group_by_filter(["district"])
            .do()
        )
        
        districts = []
        groups = response.get("data", {}).get("Aggregate", {}).get("Fact", [])
        for group in groups:
            district = group.get("groupedBy", {}).get("value")
            if district:
                districts.append(district)
        
        return {"districts": sorted(districts)}
        
    except Exception as e:
        logger.error(f"Districts query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Districts query failed: {str(e)}")

@app.get("/indicators")
async def get_indicators():
    """Get list of available indicators"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        response = (
            rag_system.weaviate_client.query
            .aggregate("Fact")
            .with_group_by_filter(["indicator"])
            .do()
        )
        
        indicators = []
        groups = response.get("data", {}).get("Aggregate", {}).get("Fact", [])
        for group in groups:
            indicator = group.get("groupedBy", {}).get("value")
            if indicator:
                indicators.append(indicator)
        
        return {"indicators": sorted(indicators)}
        
    except Exception as e:
        logger.error(f"Indicators query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indicators query failed: {str(e)}")

@app.get("/bridge/{bridge_id}")
async def get_bridge_details(bridge_id: str):
    """Get details of a specific bridge table"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        response = (
            rag_system.weaviate_client.query
            .get("BridgeTable", [
                "bridge_id", "bridge_type", "key_district", "key_indicator", "key_year",
                "connected_facts", "summary_stats", "indicators_covered", "metadata"
            ])
            .with_where({
                "path": ["bridge_id"],
                "operator": "Equal",
                "valueText": bridge_id
            })
            .with_limit(1)
            .do()
        )
        
        bridges = response.get("data", {}).get("Get", {}).get("BridgeTable", [])
        
        if not bridges:
            raise HTTPException(status_code=404, detail="Bridge not found")
        
        return bridges[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bridge query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bridge query failed: {str(e)}")

def main():
    """Run the API server"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting Hybrid RAG API Server")
    
    uvicorn.run(
        "hybrid_rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()