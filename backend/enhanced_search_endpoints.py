#!/usr/bin/env python3
"""
Enhanced Search Endpoints - Advanced Search API Extensions
Provides specialized search endpoints with enhanced capabilities
"""
from typing import List, Dict, Any, Optional, Union
import logging
import asyncio
from datetime import datetime
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
import json

from backend.retriever import WeaviateRetriever
from backend.graph_manager import GraphManager
from backend.advanced_rag_api import AdvancedRAGSystem
from backend.hybrid_rag_api import HybridRAGSystem

logger = logging.getLogger(__name__)

# Pydantic models for request/response
class EnhancedSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    search_type: str = Field("intelligent", description="Type of search to perform")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    options: Optional[Dict[str, Any]] = Field(None, description="Search options")

class SearchAnalysis(BaseModel):
    query: str
    detected_entities: List[str]
    query_type: str
    complexity_score: float
    recommended_methods: List[str]
    confidence: float

class EnhancedSearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    analysis: SearchAnalysis
    metadata: Dict[str, Any]
    processing_time: float

class ComparisonRequest(BaseModel):
    queries: List[str] = Field(..., description="List of queries to compare")
    comparison_type: str = Field("semantic", description="Type of comparison")
    limit_per_query: int = Field(5, ge=1, le=20, description="Results per query")

class TrendAnalysisRequest(BaseModel):
    indicator: str = Field(..., description="Indicator to analyze")
    district: Optional[str] = Field(None, description="Specific district")
    year_range: Optional[List[int]] = Field(None, description="Year range [start, end]")
    analysis_type: str = Field("temporal", description="Type of trend analysis")

# Initialize router
router = APIRouter(prefix="/enhanced", tags=["Enhanced Search"])

# Initialize systems
retriever = WeaviateRetriever()
graph_manager = GraphManager()
advanced_rag = AdvancedRAGSystem()
hybrid_rag = HybridRAGSystem()

class EnhancedSearchEngine:
    """Enhanced search engine with intelligent query processing"""
    
    def __init__(self):
        self.retriever = retriever
        self.graph_manager = graph_manager
        self.advanced_rag = advanced_rag
        self.hybrid_rag = hybrid_rag
        
        # Search type mappings
        self.search_types = {
            'intelligent': self._intelligent_search,
            'multi_modal': self._multi_modal_search,
            'contextual': self._contextual_search,
            'exploratory': self._exploratory_search,
            'precision': self._precision_search,
            'comprehensive': self._comprehensive_search
        }
        
        logger.info("Enhanced Search Engine initialized")
    
    async def enhanced_search(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "intelligent",
        filters: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> EnhancedSearchResponse:
        """Perform enhanced search with intelligent query analysis"""
        start_time = datetime.now()
        
        try:
            # Step 1: Analyze the query
            analysis = self._analyze_query_enhanced(query)
            
            # Step 2: Select and execute search strategy
            search_func = self.search_types.get(search_type, self._intelligent_search)
            results = await search_func(query, limit, filters, options, analysis)
            
            # Step 3: Post-process and enhance results
            enhanced_results = self._enhance_results(results, analysis)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnhancedSearchResponse(
                query=query,
                results=enhanced_results,
                analysis=analysis,
                metadata={
                    'search_type': search_type,
                    'filters_applied': filters or {},
                    'total_candidates': len(results),
                    'processing_stages': ['query_analysis', 'search_execution', 'result_enhancement'],
                    'search_strategy': analysis.recommended_methods
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    def _analyze_query_enhanced(self, query: str) -> SearchAnalysis:
        """Enhanced query analysis with entity detection and intent classification"""
        
        query_lower = query.lower()
        
        # Entity detection
        entities = []
        
        # AP Districts
        ap_districts = [
            'anantapur', 'chittoor', 'east godavari', 'guntur', 'krishna',
            'kurnool', 'nellore', 'prakasam', 'srikakulam', 'visakhapatnam',
            'vizianagaram', 'west godavari', 'kadapa', 'tirupati'
        ]
        
        # Education indicators
        education_indicators = [
            'enrollment', 'schools', 'teachers', 'students', 'infrastructure',
            'budget', 'dropout', 'attendance', 'performance', 'literacy',
            'facilities', 'classrooms', 'libraries', 'laboratories', 'toilets'
        ]
        
        # Temporal indicators
        temporal_words = ['trend', 'over time', 'change', 'growth', 'decline', 'increase', 'decrease']
        
        # Comparison indicators
        comparison_words = ['compare', 'versus', 'vs', 'difference', 'between', 'among']
        
        for district in ap_districts:
            if district in query_lower:
                entities.append(f"district:{district}")
        
        for indicator in education_indicators:
            if indicator in query_lower:
                entities.append(f"indicator:{indicator}")
        
        # Query type classification
        query_type = "general"
        if any(word in query_lower for word in comparison_words):
            query_type = "comparison"
        elif any(word in query_lower for word in temporal_words):
            query_type = "temporal"
        elif len(entities) > 0:
            query_type = "entity_specific"
        elif len(query.split()) > 10:
            query_type = "complex"
        elif any(char.isdigit() for char in query):
            query_type = "statistical"
        
        # Complexity scoring
        complexity_factors = [
            len(query.split()) / 20,  # Length factor
            len(entities) / 10,       # Entity factor
            1 if any(word in query_lower for word in comparison_words) else 0,
            1 if any(word in query_lower for word in temporal_words) else 0
        ]
        complexity_score = min(sum(complexity_factors) / len(complexity_factors), 1.0)
        
        # Recommend search methods based on analysis
        recommended_methods = []
        if query_type == "comparison":
            recommended_methods = ["advanced", "hybrid", "vector"]
        elif query_type == "temporal":
            recommended_methods = ["graph", "bridge", "vector"]
        elif query_type == "entity_specific":
            recommended_methods = ["bridge", "graph", "hybrid"]
        elif query_type == "complex":
            recommended_methods = ["advanced", "hybrid"]
        else:
            recommended_methods = ["vector", "hybrid"]
        
        # Confidence based on entity detection and query clarity
        confidence = min(0.5 + (len(entities) * 0.2) + (0.3 if query_type != "general" else 0), 1.0)
        
        return SearchAnalysis(
            query=query,
            detected_entities=entities,
            query_type=query_type,
            complexity_score=complexity_score,
            recommended_methods=recommended_methods,
            confidence=confidence
        )
    
    async def _intelligent_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        options: Optional[Dict[str, Any]],
        analysis: SearchAnalysis
    ) -> List[Dict[str, Any]]:
        """Intelligent search that adapts to query characteristics"""
        
        if analysis.complexity_score > 0.7:
            # Use advanced RAG for complex queries
            result = await self.advanced_rag.advanced_search(
                query, limit, enable_reranking=True, enable_context_expansion=True
            )
            return result.get('results', [])
        elif len(analysis.recommended_methods) > 2:
            # Use hybrid search for multi-faceted queries
            result = await self.hybrid_rag.hybrid_search(
                query, limit, fusion_strategy='adaptive_weights'
            )
            return result.get('results', [])
        else:
            # Use simple vector search for straightforward queries
            return self.retriever.vector_search(query, limit)
    
    async def _multi_modal_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        options: Optional[Dict[str, Any]],
        analysis: SearchAnalysis
    ) -> List[Dict[str, Any]]:
        """Multi-modal search combining different search paradigms"""
        
        # Execute multiple search modes concurrently
        tasks = [
            self._execute_vector_search(query, limit // 3),
            self._execute_graph_search(query, limit // 3),
            self._execute_keyword_search(query, limit // 3)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and deduplicate results
        combined_results = []
        seen_ids = set()
        
        for result_set in results:
            if isinstance(result_set, Exception):
                logger.warning(f"Multi-modal search component failed: {result_set}")
                continue
            
            for result in result_set:
                fact_id = result.get('fact_id')
                if fact_id and fact_id not in seen_ids:
                    seen_ids.add(fact_id)
                    combined_results.append(result)
        
        return combined_results[:limit]
    
    async def _contextual_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        options: Optional[Dict[str, Any]],
        analysis: SearchAnalysis
    ) -> List[Dict[str, Any]]:
        """Contextual search that expands based on entity relationships"""
        
        # Get initial results
        initial_results = self.retriever.vector_search(query, limit)
        
        # Expand context for each result
        expanded_results = []
        for result in initial_results:
            fact_id = result.get('fact_id')
            if fact_id:
                # Get related context
                context = self.graph_manager.get_entity_context([fact_id], depth=2)
                result['expanded_context'] = context
            
            expanded_results.append(result)
        
        return expanded_results
    
    async def _exploratory_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        options: Optional[Dict[str, Any]],
        analysis: SearchAnalysis
    ) -> List[Dict[str, Any]]:
        """Exploratory search for discovering related concepts"""
        
        # Use advanced RAG with high diversity
        result = await self.advanced_rag.advanced_search(
            query, limit * 2, enable_context_expansion=True, diversity_threshold=0.5
        )
        
        return result.get('results', [])[:limit]
    
    async def _precision_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        options: Optional[Dict[str, Any]],
        analysis: SearchAnalysis
    ) -> List[Dict[str, Any]]:
        """Precision search for exact matches"""
        
        # Use keyword search with high threshold
        results = self.retriever.keyword_search(query, limit)
        
        # Filter by high confidence threshold
        precision_threshold = options.get('precision_threshold', 0.8) if options else 0.8
        filtered_results = [
            r for r in results 
            if r.get('score', 0) >= precision_threshold
        ]
        
        return filtered_results
    
    async def _comprehensive_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        options: Optional[Dict[str, Any]],
        analysis: SearchAnalysis
    ) -> List[Dict[str, Any]]:
        """Comprehensive search using all available methods"""
        
        result = await self.hybrid_rag.hybrid_search(
            query, limit, fusion_strategy='ensemble_voting'
        )
        
        return result.get('results', [])
    
    async def _execute_vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Execute vector search"""
        try:
            return self.retriever.vector_search(query, limit)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _execute_graph_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Execute graph search"""
        try:
            return self.graph_manager.graph_search(query, limit)
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    async def _execute_keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Execute keyword search"""
        try:
            return self.retriever.keyword_search(query, limit)
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _enhance_results(self, results: List[Dict[str, Any]], analysis: SearchAnalysis) -> List[Dict[str, Any]]:
        """Enhance results with additional metadata and explanations"""
        
        enhanced_results = []
        
        for i, result in enumerate(results):
            # Add ranking information
            result['enhanced_rank'] = i + 1
            result['query_relevance'] = self._calculate_query_relevance(result, analysis)
            result['result_explanation'] = self._generate_result_explanation(result, analysis)
            
            # Add entity matching information
            result['entity_matches'] = self._find_entity_matches(result, analysis.detected_entities)
            
            enhanced_results.append(result)
        
        return enhanced_results
    
    def _calculate_query_relevance(self, result: Dict[str, Any], analysis: SearchAnalysis) -> float:
        """Calculate how relevant this result is to the query"""
        
        relevance_score = result.get('score', 0.5)
        
        # Boost for entity matches
        entity_boost = 0
        for entity in analysis.detected_entities:
            if entity.startswith('district:'):
                district = entity.replace('district:', '')
                if result.get('district', '').lower() == district:
                    entity_boost += 0.2
            elif entity.startswith('indicator:'):
                indicator = entity.replace('indicator:', '')
                if indicator in result.get('indicator', '').lower():
                    entity_boost += 0.2
        
        # Query type relevance
        type_boost = 0
        if analysis.query_type == 'statistical' and any(char.isdigit() for char in str(result.get('value', ''))):
            type_boost = 0.1
        elif analysis.query_type == 'temporal' and result.get('year'):
            type_boost = 0.1
        
        return min(relevance_score + entity_boost + type_boost, 1.0)
    
    def _generate_result_explanation(self, result: Dict[str, Any], analysis: SearchAnalysis) -> str:
        """Generate explanation for why this result was selected"""
        
        explanations = []
        
        # Relevance explanation
        relevance = result.get('query_relevance', 0)
        if relevance > 0.8:
            explanations.append("Highly relevant to your query")
        elif relevance > 0.6:
            explanations.append("Good match for your query")
        else:
            explanations.append("Related to your search terms")
        
        # Entity matching
        entity_matches = result.get('entity_matches', [])
        if entity_matches:
            explanations.append(f"Matches {len(entity_matches)} detected entities")
        
        # Query type specific explanations
        if analysis.query_type == 'comparison':
            explanations.append("Suitable for comparison analysis")
        elif analysis.query_type == 'temporal':
            explanations.append("Contains temporal/trend information")
        elif analysis.query_type == 'statistical':
            explanations.append("Contains statistical data")
        
        return ". ".join(explanations) + "."
    
    def _find_entity_matches(self, result: Dict[str, Any], detected_entities: List[str]) -> List[str]:
        """Find which detected entities match this result"""
        
        matches = []
        
        for entity in detected_entities:
            if entity.startswith('district:'):
                district = entity.replace('district:', '')
                if result.get('district', '').lower() == district:
                    matches.append(entity)
            elif entity.startswith('indicator:'):
                indicator = entity.replace('indicator:', '')
                if indicator in result.get('indicator', '').lower():
                    matches.append(entity)
        
        return matches

# Initialize the enhanced search engine
enhanced_engine = EnhancedSearchEngine()

# API Endpoints

@router.post("/search", response_model=EnhancedSearchResponse)
async def enhanced_search_endpoint(request: EnhancedSearchRequest):
    """Enhanced search with intelligent query processing"""
    return await enhanced_engine.enhanced_search(
        query=request.query,
        limit=request.limit,
        search_type=request.search_type,
        filters=request.filters,
        options=request.options
    )

@router.post("/compare")
async def compare_queries(request: ComparisonRequest):
    """Compare multiple queries and their results"""
    start_time = datetime.now()
    
    try:
        comparison_results = {}
        
        for query in request.queries:
            result = await enhanced_engine.enhanced_search(
                query=query,
                limit=request.limit_per_query,
                search_type="intelligent"
            )
            comparison_results[query] = result
        
        # Analyze similarities and differences
        analysis = _analyze_query_similarities(request.queries, comparison_results)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "comparison_type": request.comparison_type,
            "queries": request.queries,
            "results": comparison_results,
            "analysis": analysis,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Query comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.post("/trend-analysis")
async def trend_analysis(request: TrendAnalysisRequest):
    """Perform trend analysis for specific indicators"""
    start_time = datetime.now()
    
    try:
        # Build query for trend analysis
        query = f"{request.indicator}"
        if request.district:
            query += f" in {request.district}"
        
        # Search for historical data
        result = await enhanced_engine.enhanced_search(
            query=query,
            limit=50,  # Get more results for trend analysis
            search_type="comprehensive"
        )
        
        # Analyze trends
        trend_data = _analyze_trends(result.results, request)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "indicator": request.indicator,
            "district": request.district,
            "year_range": request.year_range,
            "analysis_type": request.analysis_type,
            "trend_data": trend_data,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")

@router.get("/search-types")
async def get_search_types():
    """Get available search types and their descriptions"""
    return {
        "intelligent": "Adapts search strategy based on query analysis",
        "multi_modal": "Combines vector, graph, and keyword search",
        "contextual": "Expands results with entity relationships",
        "exploratory": "Discovers related concepts with high diversity",
        "precision": "Focuses on exact matches with high confidence",
        "comprehensive": "Uses all available search methods with ensemble voting"
    }

@router.get("/capabilities")
async def get_capabilities():
    """Get enhanced search engine capabilities"""
    return {
        "features": [
            "Intelligent query analysis",
            "Entity detection and classification",
            "Adaptive search strategy selection",
            "Multi-modal search execution",
            "Result enhancement and explanation",
            "Query comparison analysis",
            "Trend analysis capabilities",
            "Contextual result expansion"
        ],
        "supported_entities": [
            "AP districts",
            "Education indicators", 
            "Temporal expressions",
            "Statistical terms"
        ],
        "query_types": [
            "general",
            "comparison",
            "temporal",
            "entity_specific",
            "complex",
            "statistical"
        ]
    }

# Utility functions

def _analyze_query_similarities(queries: List[str], results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze similarities between queries and their results"""
    
    # Simple similarity analysis based on common results
    common_results = set()
    all_fact_ids = []
    
    for query, result in results.items():
        query_fact_ids = set(r.get('fact_id') for r in result.results if r.get('fact_id'))
        all_fact_ids.append(query_fact_ids)
    
    # Find intersection of all queries
    if all_fact_ids:
        common_results = set.intersection(*all_fact_ids)
    
    return {
        "total_queries": len(queries),
        "common_results": len(common_results),
        "similarity_score": len(common_results) / max(len(queries), 1),
        "unique_results_per_query": {
            query: len(set(r.get('fact_id') for r in result.results if r.get('fact_id')) - common_results)
            for query, result in results.items()
        }
    }

def _analyze_trends(results: List[Dict[str, Any]], request: TrendAnalysisRequest) -> Dict[str, Any]:
    """Analyze trends in the search results"""
    
    # Group results by year
    yearly_data = {}
    
    for result in results:
        year = result.get('year')
        value = result.get('value')
        
        if year and value:
            try:
                year = int(year)
                if request.year_range:
                    if year < request.year_range[0] or year > request.year_range[1]:
                        continue
                
                if year not in yearly_data:
                    yearly_data[year] = []
                
                # Try to convert value to numeric
                try:
                    numeric_value = float(str(value).replace(',', ''))
                    yearly_data[year].append(numeric_value)
                except:
                    pass
                    
            except:
                continue
    
    # Calculate trend statistics
    trend_stats = {}
    sorted_years = sorted(yearly_data.keys())
    
    for year in sorted_years:
        values = yearly_data[year]
        if values:
            trend_stats[year] = {
                'count': len(values),
                'average': sum(values) / len(values),
                'total': sum(values),
                'min': min(values),
                'max': max(values)
            }
    
    # Calculate overall trend direction
    trend_direction = "stable"
    if len(trend_stats) >= 2:
        first_year_avg = list(trend_stats.values())[0]['average']
        last_year_avg = list(trend_stats.values())[-1]['average']
        
        change_pct = ((last_year_avg - first_year_avg) / first_year_avg) * 100
        
        if change_pct > 5:
            trend_direction = "increasing"
        elif change_pct < -5:
            trend_direction = "decreasing"
    
    return {
        "yearly_data": trend_stats,
        "trend_direction": trend_direction,
        "data_points": sum(len(values) for values in yearly_data.values()),
        "year_coverage": len(sorted_years),
        "year_range_actual": [min(sorted_years), max(sorted_years)] if sorted_years else None
    }