#!/usr/bin/env python3
"""
Hybrid RAG System: Graph RAG + Vector RAG + Bridge Tables
Combines Weaviate vector search, Neo4j graph traversal, and bridge table connections
"""
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import json
import re
import weaviate
from neo4j import GraphDatabase
import numpy as np
from sentence_transformers import SentenceTransformer
from bridge_table_manager import BridgeTableManager

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Unified search result from hybrid RAG"""
    fact_id: str
    content: str
    district: str
    indicator: str
    year: int
    value: float
    unit: str
    confidence_score: float
    source: str  # 'vector', 'graph', 'bridge'
    graph_context: Optional[Dict[str, Any]] = None
    bridge_connections: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class HybridSearchQuery:
    """Query structure for hybrid search"""
    query_text: str
    district_filter: Optional[str] = None
    indicator_filter: Optional[str] = None
    year_filter: Optional[int] = None
    year_range: Optional[Tuple[int, int]] = None
    search_depth: int = 2  # Graph traversal depth
    include_graph_context: bool = True
    include_bridge_expansion: bool = True
    max_results: int = 10

class HybridRAGSystem:
    """Hybrid RAG system combining vector search, graph traversal, and bridge tables"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize vector search (Weaviate)
        self.weaviate_client = self._init_weaviate()
        
        # Initialize graph database (Neo4j)
        self.neo4j_driver = self._init_neo4j()
        
        # Initialize bridge table manager
        self.bridge_manager = BridgeTableManager(config.get('data_dir', 'data'))
        self.bridge_manager.load_bridge_tables()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        # Search configuration
        self.search_config = {
            'vector_weight': 0.4,      # Weight for vector search results
            'graph_weight': 0.3,       # Weight for graph search results  
            'bridge_weight': 0.3,      # Weight for bridge table results
            'similarity_threshold': 0.7,
            'graph_max_depth': 3,
            'bridge_max_connections': 15
        }
        
        logger.info("🤖 Hybrid RAG System initialized")
    
    def search(self, query: HybridSearchQuery) -> List[SearchResult]:
        """Main hybrid search function"""
        logger.info(f"🔍 Hybrid search: '{query.query_text[:50]}...'")
        
        all_results = []
        
        # Stage 1: Vector Search (Weaviate)
        vector_results = self._vector_search(query)
        all_results.extend(vector_results)
        logger.info(f"📊 Vector search found {len(vector_results)} results")
        
        # Stage 2: Bridge Table Expansion
        if query.include_bridge_expansion:
            bridge_results = self._bridge_table_search(query, vector_results)
            all_results.extend(bridge_results)
            logger.info(f"🌉 Bridge search found {len(bridge_results)} additional results")
        
        # Stage 3: Graph Context Expansion
        if query.include_graph_context:
            graph_results = self._graph_search(query, all_results)
            all_results.extend(graph_results)
            logger.info(f"🕸️ Graph search found {len(graph_results)} additional results")
        
        # Stage 4: Fusion and Re-ranking
        final_results = self._fuse_and_rerank(all_results, query)
        
        logger.info(f"✅ Hybrid search completed: {len(final_results)} final results")
        return final_results[:query.max_results]
    
    def _vector_search(self, query: HybridSearchQuery) -> List[SearchResult]:
        """Perform vector search using Weaviate"""
        try:
            # Build Weaviate query
            where_conditions = []
            
            if query.district_filter:
                where_conditions.append({
                    "path": ["district"],
                    "operator": "Equal",
                    "valueText": query.district_filter
                })
            
            if query.indicator_filter:
                where_conditions.append({
                    "path": ["indicator"],
                    "operator": "Equal", 
                    "valueText": query.indicator_filter
                })
            
            if query.year_filter:
                where_conditions.append({
                    "path": ["year"],
                    "operator": "Equal",
                    "valueInt": query.year_filter
                })
            elif query.year_range:
                where_conditions.append({
                    "path": ["year"],
                    "operator": "GreaterThanEqual",
                    "valueInt": query.year_range[0]
                })
                where_conditions.append({
                    "path": ["year"],
                    "operator": "LessThanEqual", 
                    "valueInt": query.year_range[1]
                })
            
            # Combine conditions
            where_clause = None
            if where_conditions:
                if len(where_conditions) == 1:
                    where_clause = where_conditions[0]
                else:
                    where_clause = {
                        "operator": "And",
                        "operands": where_conditions
                    }
            
            # Perform hybrid search (vector + BM25)
            response = (
                self.weaviate_client.query
                .get("Fact", [
                    "fact_id", "district", "indicator", "year", "value", "unit",
                    "content", "source_document", "confidence_score", "metadata"
                ])
                .with_hybrid(
                    query=query.query_text,
                    alpha=0.75  # 0.75 = more vector, 0.25 = more BM25
                )
                .with_where(where_clause)
                .with_limit(query.max_results * 2)  # Get more for fusion
                .with_additional(["score"])
                .do()
            )
            
            # Convert to SearchResult objects
            results = []
            facts = response.get("data", {}).get("Get", {}).get("Fact", [])
            
            for fact in facts:
                result = SearchResult(
                    fact_id=fact.get("fact_id", ""),
                    content=fact.get("content", ""),
                    district=fact.get("district", ""),
                    indicator=fact.get("indicator", ""),
                    year=fact.get("year", 2023),
                    value=fact.get("value", 0.0),
                    unit=fact.get("unit", ""),
                    confidence_score=fact.get("_additional", {}).get("score", 0.0),
                    source="vector",
                    metadata=fact.get("metadata", {})
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _bridge_table_search(self, query: HybridSearchQuery, initial_results: List[SearchResult]) -> List[SearchResult]:
        """Expand search using bridge tables"""
        try:
            bridge_results = []
            
            # Extract query context from initial results and query
            query_context = self._extract_query_context(query, initial_results)
            
            # Get related fact IDs from bridge tables
            related_fact_ids = self.bridge_manager.get_related_facts(
                query_context, 
                max_results=self.search_config['bridge_max_connections']
            )
            
            # Filter out facts we already have
            existing_fact_ids = {result.fact_id for result in initial_results}
            new_fact_ids = [fid for fid in related_fact_ids if fid not in existing_fact_ids]
            
            # Retrieve full fact data for new IDs
            if new_fact_ids:
                bridge_facts = self._retrieve_facts_by_ids(new_fact_ids)
                
                for fact in bridge_facts:
                    result = SearchResult(
                        fact_id=fact.get("fact_id", ""),
                        content=fact.get("content", ""),
                        district=fact.get("district", ""),
                        indicator=fact.get("indicator", ""),
                        year=fact.get("year", 2023),
                        value=fact.get("value", 0.0),
                        unit=fact.get("unit", ""),
                        confidence_score=0.8,  # Bridge connections have good confidence
                        source="bridge",
                        bridge_connections=related_fact_ids,
                        metadata=fact.get("metadata", {})
                    )
                    bridge_results.append(result)
            
            return bridge_results
            
        except Exception as e:
            logger.error(f"Bridge table search failed: {e}")
            return []
    
    def _graph_search(self, query: HybridSearchQuery, existing_results: List[SearchResult]) -> List[SearchResult]:
        """Expand search using Neo4j graph traversal"""
        try:
            graph_results = []
            
            # Extract seed nodes from existing results
            seed_fact_ids = [result.fact_id for result in existing_results[:5]]  # Use top 5 as seeds
            
            if not seed_fact_ids:
                return []
            
            # Build Cypher query for graph expansion
            cypher_query = """
            MATCH (seed:Fact)
            WHERE seed.fact_id IN $seed_ids
            
            // Multi-hop traversal to find related facts
            MATCH path = (seed)-[:RELATED_TO|:SAME_DISTRICT|:SAME_INDICATOR|:TEMPORAL_NEXT*1..%d]-(related:Fact)
            
            // Apply filters
            WHERE related.fact_id IS NOT NULL
            %s
            
            // Calculate path-based relevance score
            WITH related, seed, path,
                 CASE 
                     WHEN length(path) = 1 THEN 1.0
                     WHEN length(path) = 2 THEN 0.8  
                     WHEN length(path) = 3 THEN 0.6
                     ELSE 0.4
                 END as path_score
            
            // Return related facts with context
            RETURN DISTINCT 
                related.fact_id as fact_id,
                related.district as district,
                related.indicator as indicator, 
                related.year as year,
                related.value as value,
                related.unit as unit,
                related.content as content,
                related.source_document as source_document,
                related.confidence_score as confidence_score,
                path_score,
                collect(DISTINCT seed.fact_id) as connected_seeds,
                length(path) as graph_distance
            
            ORDER BY path_score DESC, related.confidence_score DESC
            LIMIT %d
            """ % (
                query.search_depth,
                self._build_graph_filters(query),
                query.max_results * 2
            )
            
            # Execute query
            with self.neo4j_driver.session() as session:
                result = session.run(cypher_query, seed_ids=seed_fact_ids)
                
                for record in result:
                    # Skip if we already have this fact
                    fact_id = record["fact_id"]
                    if fact_id in {r.fact_id for r in existing_results}:
                        continue
                    
                    graph_result = SearchResult(
                        fact_id=fact_id,
                        content=record.get("content", ""),
                        district=record.get("district", ""),
                        indicator=record.get("indicator", ""),
                        year=record.get("year", 2023),
                        value=record.get("value", 0.0),
                        unit=record.get("unit", ""),
                        confidence_score=record.get("path_score", 0.5),
                        source="graph",
                        graph_context={
                            "connected_seeds": record.get("connected_seeds", []),
                            "graph_distance": record.get("graph_distance", 1),
                            "path_score": record.get("path_score", 0.5)
                        },
                        metadata={}
                    )
                    graph_results.append(graph_result)
            
            return graph_results
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    def _fuse_and_rerank(self, all_results: List[SearchResult], query: HybridSearchQuery) -> List[SearchResult]:
        """Fuse results from different sources and re-rank"""
        try:
            # Remove duplicates (by fact_id)
            unique_results = {}
            for result in all_results:
                if result.fact_id not in unique_results:
                    unique_results[result.fact_id] = result
                else:
                    # Merge information from duplicate results
                    existing = unique_results[result.fact_id]
                    if result.confidence_score > existing.confidence_score:
                        existing.confidence_score = result.confidence_score
                    
                    # Combine sources
                    if result.source not in existing.source:
                        existing.source = f"{existing.source}+{result.source}"
                    
                    # Merge graph context
                    if result.graph_context and not existing.graph_context:
                        existing.graph_context = result.graph_context
                    
                    # Merge bridge connections
                    if result.bridge_connections and not existing.bridge_connections:
                        existing.bridge_connections = result.bridge_connections
            
            results = list(unique_results.values())
            
            # Calculate fusion scores
            for result in results:
                fusion_score = self._calculate_fusion_score(result, query)
                result.confidence_score = fusion_score
            
            # Sort by fusion score
            results.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            return all_results
    
    def _calculate_fusion_score(self, result: SearchResult, query: HybridSearchQuery) -> float:
        """Calculate final fusion score for result"""
        base_score = result.confidence_score
        
        # Source-based weighting
        source_multiplier = 1.0
        if "vector" in result.source:
            source_multiplier *= self.search_config['vector_weight']
        if "graph" in result.source:
            source_multiplier *= self.search_config['graph_weight']
        if "bridge" in result.source:
            source_multiplier *= self.search_config['bridge_weight']
        
        # Multiple source bonus
        source_count = len(result.source.split('+'))
        multi_source_bonus = min(0.2, (source_count - 1) * 0.1)
        
        # Filter match bonus
        filter_bonus = 0.0
        if query.district_filter and result.district == query.district_filter:
            filter_bonus += 0.1
        if query.indicator_filter and result.indicator == query.indicator_filter:
            filter_bonus += 0.1
        if query.year_filter and result.year == query.year_filter:
            filter_bonus += 0.1
        
        # Year recency bonus (more recent = slightly better)
        current_year = 2023
        year_bonus = max(0, (result.year - 2000) / (current_year - 2000)) * 0.05
        
        # Graph context bonus
        graph_bonus = 0.0
        if result.graph_context:
            graph_distance = result.graph_context.get("graph_distance", 1)
            graph_bonus = max(0, (4 - graph_distance) / 4) * 0.1
        
        # Bridge connection bonus
        bridge_bonus = 0.0
        if result.bridge_connections:
            bridge_bonus = min(0.1, len(result.bridge_connections) / 10 * 0.1)
        
        # Calculate final score
        final_score = (
            base_score * source_multiplier + 
            multi_source_bonus + 
            filter_bonus + 
            year_bonus + 
            graph_bonus + 
            bridge_bonus
        )
        
        return min(1.0, final_score)  # Cap at 1.0
    
    def _extract_query_context(self, query: HybridSearchQuery, results: List[SearchResult]) -> Dict[str, Any]:
        """Extract context from query and initial results for bridge table lookup"""
        context = {}
        
        # From query filters
        if query.district_filter:
            context['district'] = query.district_filter
        if query.indicator_filter:
            context['indicator'] = query.indicator_filter
        if query.year_filter:
            context['year'] = query.year_filter
        
        # From initial results
        if results:
            # Most common district in results
            districts = [r.district for r in results if r.district]
            if districts and not context.get('district'):
                context['district'] = max(set(districts), key=districts.count)
            
            # Most common indicator in results
            indicators = [r.indicator for r in results if r.indicator]
            if indicators and not context.get('indicator'):
                context['indicator'] = max(set(indicators), key=indicators.count)
            
            # Most recent year in results
            years = [r.year for r in results if r.year]
            if years and not context.get('year'):
                context['year'] = max(years)
        
        return context
    
    def _retrieve_facts_by_ids(self, fact_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve full fact data by IDs from Weaviate"""
        try:
            if not fact_ids:
                return []
            
            # Query Weaviate for facts by ID
            response = (
                self.weaviate_client.query
                .get("Fact", [
                    "fact_id", "district", "indicator", "year", "value", "unit",
                    "content", "source_document", "confidence_score", "metadata"
                ])
                .with_where({
                    "path": ["fact_id"],
                    "operator": "ContainsAny",
                    "valueText": fact_ids
                })
                .with_limit(len(fact_ids))
                .do()
            )
            
            return response.get("data", {}).get("Get", {}).get("Fact", [])
            
        except Exception as e:
            logger.error(f"Failed to retrieve facts by IDs: {e}")
            return []
    
    def _build_graph_filters(self, query: HybridSearchQuery) -> str:
        """Build Cypher WHERE clause for graph filters"""
        conditions = []
        
        if query.district_filter:
            conditions.append(f"related.district = '{query.district_filter}'")
        
        if query.indicator_filter:
            conditions.append(f"related.indicator = '{query.indicator_filter}'")
        
        if query.year_filter:
            conditions.append(f"related.year = {query.year_filter}")
        elif query.year_range:
            conditions.append(
                f"related.year >= {query.year_range[0]} AND related.year <= {query.year_range[1]}"
            )
        
        if conditions:
            return "AND " + " AND ".join(conditions)
        else:
            return ""
    
    def _init_weaviate(self) -> weaviate.Client:
        """Initialize Weaviate client"""
        try:
            client = weaviate.Client(
                url=self.config.get('weaviate_url', 'http://localhost:8080'),
                timeout_config=(5, 15)
            )
            
            # Test connection
            if not client.is_ready():
                raise ConnectionError("Weaviate is not ready")
            
            logger.info("✅ Weaviate connection established")
            return client
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    def _init_neo4j(self) -> GraphDatabase.driver:
        """Initialize Neo4j driver"""
        try:
            driver = GraphDatabase.driver(
                self.config.get('neo4j_uri', 'bolt://localhost:7687'),
                auth=(
                    self.config.get('neo4j_user', 'neo4j'),
                    self.config.get('neo4j_password', 'password')
                )
            )
            
            # Test connection
            with driver.session() as session:
                session.run("RETURN 1")
            
            logger.info("✅ Neo4j connection established")
            return driver
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close database connections"""
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()
        
        logger.info("🔒 Hybrid RAG System connections closed")

# Convenience function for easy usage
def create_hybrid_rag_system(config_file: str = None) -> HybridRAGSystem:
    """Create and initialize hybrid RAG system"""
    
    # Default configuration
    default_config = {
        'weaviate_url': 'http://localhost:8080',
        'neo4j_uri': 'bolt://localhost:7687',
        'neo4j_user': 'neo4j',
        'neo4j_password': 'password',
        'embedding_model': 'all-MiniLM-L6-v2',
        'data_dir': 'data'
    }
    
    # Load config file if provided
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            default_config.update(file_config)
    
    return HybridRAGSystem(default_config)