#!/usr/bin/env python3
"""
Unified Retriever for AP Policy Co-Pilot
Combines vector search (Weaviate), graph reasoning (Neo4j), and legal hierarchy navigation
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
from datetime import datetime

import weaviate
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Unified retrieval result"""
    doc_id: str
    content: str
    doc_type: str  # 'legal', 'data', 'framework'
    score: float
    
    # Legal metadata
    title: Optional[str] = None
    section: Optional[str] = None
    clause: Optional[str] = None
    go_number: Optional[str] = None
    issued_date: Optional[str] = None
    
    # Data metadata
    indicator: Optional[str] = None
    district: Optional[str] = None
    year: Optional[int] = None
    value: Optional[Any] = None
    
    # Provenance
    source: Optional[str] = None
    page_number: Optional[int] = None
    
    # Retrieval metadata
    retrieval_method: str = "hybrid"
    confidence: float = 1.0

class UnifiedRetriever:
    """
    Unified retrieval engine combining:
    1. Vector search (semantic similarity)
    2. Keyword search (BM25)
    3. Graph reasoning (entity relationships)
    4. Legal hierarchy navigation (Act→Rule→GO)
    5. Temporal reasoning (year-based filtering)
    """
    
    def __init__(
        self,
        weaviate_config: Dict[str, Any],
        neo4j_config: Dict[str, Any],
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.weaviate_config = weaviate_config
        self.neo4j_config = neo4j_config
        
        # Initialize connections
        self.weaviate_client = None
        self.neo4j_driver = None
        self.embedding_model = None
        
        # Initialize components
        self._init_weaviate()
        self._init_neo4j()
        self._init_embedding_model(embedding_model)
        
        logger.info("✅ Unified Retriever initialized")
    
    def _init_weaviate(self):
        """Initialize Weaviate connection"""
        try:
            self.weaviate_client = weaviate.Client(
                url=self.weaviate_config["url"],
                timeout_config=(5, self.weaviate_config.get("timeout", 30))
            )
            logger.info(f"   ✅ Weaviate connected: {self.weaviate_config['url']}")
        except Exception as e:
            logger.error(f"   ❌ Weaviate connection failed: {e}")
            raise
    
    def _init_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_config["uri"],
                auth=(self.neo4j_config["user"], self.neo4j_config["password"])
            )
            # Test connection
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            logger.info(f"   ✅ Neo4j connected: {self.neo4j_config['uri']}")
        except Exception as e:
            logger.error(f"   ❌ Neo4j connection failed: {e}")
            raise
    
    def _init_embedding_model(self, model_name: str):
        """Initialize sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"   ✅ Embedding model loaded: {model_name}")
        except Exception as e:
            logger.error(f"   ❌ Embedding model loading failed: {e}")
            raise
    
    async def retrieve_legal(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve legal documents (Acts, Rules, GOs)"""
        logger.info(f"Retrieving legal documents for: '{query}'")
        
        # Build Weaviate query with filters
        where_filter = {
            "operator": "And",
            "operands": [
                {
                    "path": ["doc_type"],
                    "operator": "Equal",
                    "valueText": "legal"
                }
            ]
        }
        
        # Add entity filters
        if filters and filters.get('entities'):
            for entity in filters['entities']:
                if entity.startswith('YEAR:'):
                    year = entity.split(':')[1]
                    where_filter["operands"].append({
                        "path": ["issued_date"],
                        "operator": "Like",
                        "valueText": f"*{year}*"
                    })
        
        try:
            # Hybrid search: vector + keyword
            result = (
                self.weaviate_client.query
                .get("PolicyDocument", [
                    "doc_id", "title", "doc_type", "full_text", 
                    "section", "clause", "go_number", "issued_date",
                    "source_file", "page_number", "confidence"
                ])
                .with_hybrid(
                    query=query,
                    alpha=0.7  # 70% vector, 30% keyword
                )
                .with_where(where_filter)
                .with_limit(max_results)
                .do()
            )
            
            documents = result.get("data", {}).get("Get", {}).get("PolicyDocument", [])
            logger.info(f"   Found {len(documents)} legal documents")
            
            return documents
            
        except Exception as e:
            logger.error(f"   ❌ Legal retrieval failed: {e}")
            return []
    
    async def retrieve_data(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve statistical data points"""
        logger.info(f"Retrieving data for: '{query}'")
        
        # Build filter for data documents
        where_filter = {
            "operator": "Or",
            "operands": [
                {
                    "path": ["source"],
                    "operator": "Equal",
                    "valueText": "udise"
                },
                {
                    "path": ["source"],
                    "operator": "Equal",
                    "valueText": "scert"
                },
                {
                    "path": ["source"],
                    "operator": "Equal",
                    "valueText": "cse"
                }
            ]
        }
        
        # Add entity filters
        if filters and filters.get('entities'):
            for entity in filters['entities']:
                if entity.startswith('DISTRICT:'):
                    district = entity.split(':')[1]
                    where_filter["operands"].append({
                        "path": ["district"],
                        "operator": "Equal",
                        "valueText": district
                    })
                elif entity.startswith('YEAR:'):
                    year = int(entity.split(':')[1])
                    where_filter["operands"].append({
                        "path": ["year"],
                        "operator": "Equal",
                        "valueInt": year
                    })
        
        try:
            result = (
                self.weaviate_client.query
                .get("DataPoint", [
                    "data_id", "indicator", "indicator_label", "value", "unit",
                    "district", "year", "source", "source_document", "confidence"
                ])
                .with_hybrid(
                    query=query,
                    alpha=0.6  # More keyword weight for data
                )
                .with_where(where_filter)
                .with_limit(max_results)
                .do()
            )
            
            data_points = result.get("data", {}).get("Get", {}).get("DataPoint", [])
            logger.info(f"   Found {len(data_points)} data points")
            
            return data_points
            
        except Exception as e:
            logger.error(f"   ❌ Data retrieval failed: {e}")
            return []
    
    async def retrieve_with_legal_hierarchy(
        self,
        query: str,
        max_results: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve documents and organize by legal hierarchy (Act → Rule → GO)
        """
        logger.info(f"Retrieving with legal hierarchy for: '{query}'")
        
        # First, get all relevant legal documents
        legal_docs = await self.retrieve_legal(query, max_results=max_results * 2)
        
        # Organize by hierarchy
        hierarchy = {
            'acts': [],
            'rules': [],
            'gos': [],
            'other': []
        }
        
        for doc in legal_docs:
            doc_type = doc.get('doc_type', '').lower()
            if 'act' in doc_type:
                hierarchy['acts'].append(doc)
            elif 'rule' in doc_type:
                hierarchy['rules'].append(doc)
            elif 'go' in doc_type:
                hierarchy['gos'].append(doc)
            else:
                hierarchy['other'].append(doc)
        
        # Use Neo4j to find relationships
        if hierarchy['gos']:
            for go in hierarchy['gos']:
                # Find parent Rule/Act
                parents = await self._find_legal_parents(go.get('doc_id'))
                if parents:
                    go['parent_documents'] = parents
        
        logger.info(f"   Hierarchy: {len(hierarchy['acts'])} Acts, "
                   f"{len(hierarchy['rules'])} Rules, {len(hierarchy['gos'])} GOs")
        
        return hierarchy
    
    async def _find_legal_parents(self, doc_id: str) -> List[Dict[str, str]]:
        """Find parent documents in legal hierarchy using Neo4j"""
        query = """
        MATCH (child {doc_id: $doc_id})-[:IMPLEMENTS|DERIVES_FROM*1..2]->(parent)
        WHERE parent.doc_type IN ['act', 'rule']
        RETURN parent.doc_id AS doc_id, parent.title AS title, parent.doc_type AS doc_type
        ORDER BY parent.doc_type
        """
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query, doc_id=doc_id)
                parents = [dict(record) for record in result]
                return parents
        except Exception as e:
            logger.error(f"Error finding legal parents: {e}")
            return []
    
    async def retrieve_with_graph_context(
        self,
        query: str,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve documents with graph context (related entities, relationships)
        """
        logger.info(f"Retrieving with graph context for: '{query}'")
        
        # Get base results
        legal_docs = await self.retrieve_legal(query, max_results=max_results)
        data_points = await self.retrieve_data(query, max_results=max_results)
        
        # Extract entities from query
        entities = self._extract_entities(query)
        
        # Get graph context for entities
        graph_context = {}
        for entity in entities:
            context = await self._get_entity_context(entity)
            if context:
                graph_context[entity] = context
        
        return {
            'legal_documents': legal_docs,
            'data_points': data_points,
            'graph_context': graph_context,
            'entities': entities
        }
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query (simplified)"""
        entities = []
        
        # Districts
        ap_districts = [
            'Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Krishna',
            'Kurnool', 'Prakasam', 'Nellore', 'Srikakulam', 'Visakhapatnam',
            'Vizianagaram', 'West Godavari', 'YSR Kadapa'
        ]
        for district in ap_districts:
            if district.lower() in query.lower():
                entities.append(f"District:{district}")
        
        # Years
        import re
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        entities.extend([f"Year:{year}" for year in years])
        
        # Indicators
        indicators = ['dropout', 'enrollment', 'budget', 'school', 'teacher', 'infrastructure']
        for indicator in indicators:
            if indicator in query.lower():
                entities.append(f"Indicator:{indicator}")
        
        return entities
    
    async def _get_entity_context(self, entity: str) -> Dict[str, Any]:
        """Get graph context for an entity from Neo4j"""
        entity_type, entity_value = entity.split(':', 1)
        
        if entity_type == "District":
            query = """
            MATCH (d:District {name: $name})
            OPTIONAL MATCH (d)-[r]-(related)
            RETURN d, collect({type: type(r), node: related}) AS related_nodes
            LIMIT 20
            """
        elif entity_type == "Year":
            query = """
            MATCH (data:DataPoint {year: $year})
            RETURN count(data) AS data_points,
                   collect(DISTINCT data.indicator)[..10] AS indicators
            """
        elif entity_type == "Indicator":
            query = """
            MATCH (i:Indicator {name: $name})
            OPTIONAL MATCH (i)-[r]-(related)
            RETURN i, collect({type: type(r), node: related}) AS related_nodes
            LIMIT 20
            """
        else:
            return {}
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query, name=entity_value, year=int(entity_value) if entity_type == "Year" else None)
                record = result.single()
                if record:
                    return dict(record)
        except Exception as e:
            logger.error(f"Error getting entity context for {entity}: {e}")
        
        return {}
    
    async def retrieve_temporal(
        self,
        query: str,
        year_range: Optional[Tuple[int, int]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with temporal reasoning (time-series, trends)
        """
        logger.info(f"Temporal retrieval for: '{query}' (years: {year_range})")
        
        # Build temporal filter
        filters = None
        if year_range:
            filters = {
                'year_min': year_range[0],
                'year_max': year_range[1]
            }
        
        # Get data points
        data_points = await self.retrieve_data(query, max_results=max_results, filters=filters)
        
        # Sort by year
        data_points.sort(key=lambda x: x.get('year', 0))
        
        return data_points
    
    async def retrieve_comparative(
        self,
        entity1: str,
        entity2: str,
        indicator: str,
        year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieve for comparison queries (e.g., district A vs district B)
        """
        logger.info(f"Comparative retrieval: {entity1} vs {entity2} for {indicator}")
        
        # Get data for entity1
        query1 = f"{indicator} {entity1}"
        if year:
            query1 += f" {year}"
        data1 = await self.retrieve_data(query1, max_results=5)
        
        # Get data for entity2
        query2 = f"{indicator} {entity2}"
        if year:
            query2 += f" {year}"
        data2 = await self.retrieve_data(query2, max_results=5)
        
        return {
            'entity1': {
                'name': entity1,
                'data': data1
            },
            'entity2': {
                'name': entity2,
                'data': data2
            },
            'indicator': indicator,
            'year': year
        }
    
    def test_weaviate_connection(self) -> bool:
        """Test Weaviate connection"""
        try:
            self.weaviate_client.schema.get()
            return True
        except Exception as e:
            logger.error(f"Weaviate connection test failed: {e}")
            return False
    
    def test_neo4j_connection(self) -> bool:
        """Test Neo4j connection"""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            return True
        except Exception as e:
            logger.error(f"Neo4j connection test failed: {e}")
            return False
    
    def close(self):
        """Close all connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
        logger.info("Connections closed")

if __name__ == "__main__":
    # Test retriever
    async def test():
        retriever = UnifiedRetriever(
            weaviate_config={"url": "http://localhost:8080", "timeout": 30},
            neo4j_config={"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}
        )
        
        # Test legal retrieval
        legal_docs = await retriever.retrieve_legal("Nadu Nedu infrastructure", max_results=5)
        print(f"\n=== Legal Documents ({len(legal_docs)}) ===")
        for doc in legal_docs[:3]:
            print(f"- {doc.get('title')}")
        
        # Test data retrieval
        data_points = await retriever.retrieve_data("dropout rate SC students", max_results=5)
        print(f"\n=== Data Points ({len(data_points)}) ===")
        for dp in data_points[:3]:
            print(f"- {dp.get('indicator_label')}: {dp.get('value')} ({dp.get('district')}, {dp.get('year')})")
        
        # Test legal hierarchy
        hierarchy = await retriever.retrieve_with_legal_hierarchy("School Management Committee", max_results=5)
        print(f"\n=== Legal Hierarchy ===")
        print(f"Acts: {len(hierarchy['acts'])}, Rules: {len(hierarchy['rules'])}, GOs: {len(hierarchy['gos'])}")
        
        retriever.close()
    
    asyncio.run(test())