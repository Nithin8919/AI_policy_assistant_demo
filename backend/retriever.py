"""
Weaviate Retriever for AP Policy Co-Pilot
Hybrid search combining vector similarity and keyword filtering
"""
import os
import logging
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.query import MetadataQuery, Filter
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class WeaviateRetriever:
    """Hybrid retriever using Weaviate"""
    
    def __init__(self):
        # Connect to Weaviate using HTTP API (bypass gRPC issues)
        weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        self.client = weaviate.Client(url=weaviate_url)
        
        # Embedding model
        model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        
        logger.info("Initialized Weaviate retriever")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query"""
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.7  # 0.7 = 70% vector, 30% keyword
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and keyword search
        
        Args:
            query: Search query
            limit: Maximum results
            filters: Property filters (indicator, district, year, etc.)
            alpha: Balance between vector (1.0) and keyword (0.0) search
        """
        try:
            # Build where filter
            where_filter = None
            if filters:
                where_conditions = []
                
                if 'indicator' in filters:
                    where_conditions.append({
                        "path": ["indicator"],
                        "operator": "Equal",
                        "valueText": filters['indicator']
                    })
                
                if 'district' in filters:
                    where_conditions.append({
                        "path": ["district"], 
                        "operator": "Equal",
                        "valueText": filters['district']
                    })
                
                if 'year' in filters:
                    where_conditions.append({
                        "path": ["year"],
                        "operator": "Equal", 
                        "valueInt": int(filters['year'])
                    })
                
                # Combine conditions with AND
                if len(where_conditions) == 1:
                    where_filter = where_conditions[0]
                elif len(where_conditions) > 1:
                    where_filter = {
                        "operator": "And",
                        "operands": where_conditions
                    }
            
            # Determine search method based on alpha
            if alpha >= 0.9:
                # Pure vector search
                query_vector = self.generate_embedding(query)
                query_builder = (
                    self.client.query
                    .get("Fact", [
                        "fact_id", "indicator", "category", "district", "year", 
                        "value", "unit", "content", "source_document", "source_page", 
                        "confidence_score", "metadata"
                    ])
                    .with_near_vector({
                        "vector": query_vector
                    })
                )
            elif alpha <= 0.1:
                # Pure keyword search with case-insensitive matching
                query_builder = (
                    self.client.query
                    .get("Fact", [
                        "fact_id", "indicator", "category", "district", "year", 
                        "value", "unit", "content", "source_document", "source_page", 
                        "confidence_score", "metadata"
                    ])
                    .with_bm25(query=query)
                )
            else:
                # True hybrid search combining vector + BM25
                query_vector = self.generate_embedding(query)
                query_builder = (
                    self.client.query
                    .get("Fact", [
                        "fact_id", "indicator", "category", "district", "year", 
                        "value", "unit", "content", "source_document", "source_page", 
                        "confidence_score", "metadata"
                    ])
                    .with_hybrid(
                        query=query,
                        vector=query_vector,
                        alpha=alpha  # Weaviate's hybrid search with alpha parameter
                    )
                )
            
            # Add filters if provided
            if where_filter:
                query_builder = query_builder.with_where(where_filter)
            
            response = query_builder.with_additional(["score", "distance"]).with_limit(limit).do()
            
            # Format results
            results = []
            if "data" in response and "Get" in response["data"] and "Fact" in response["data"]["Get"]:
                for obj in response["data"]["Get"]["Fact"]:
                    additional = obj.get("_additional", {})
                    result = {
                        'fact_id': obj.get('fact_id'),
                        'indicator': obj.get('indicator'), 
                        'category': obj.get('category'),
                        'district': obj.get('district'),
                        'year': obj.get('year'),
                        'value': obj.get('value'),
                        'unit': obj.get('unit'),
                        'content': obj.get('content'),
                        'source': obj.get('source_document'),
                        'page_ref': obj.get('source_page'),
                        'confidence': obj.get('confidence_score'),
                        'span_text': obj.get('content'),  # Use content as span_text
                        'pdf_name': obj.get('source_document'),
                        'score': additional.get('score'),
                        'distance': additional.get('distance')
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} results for query: {query} (alpha={alpha})")
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def vector_search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Pure vector similarity search"""
        return self.search(query, limit, filters, alpha=1.0)
    
    def keyword_search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Pure keyword (BM25) search"""
        return self.search(query, limit, filters, alpha=0.0)
    
    def get_by_id(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Get fact by ID"""
        try:
            response = (
                self.client.query
                .get("Fact", [
                    "fact_id", "indicator", "district", "year", 
                    "value", "unit", "content", "source_document"
                ])
                .with_where({
                    "path": ["fact_id"],
                    "operator": "Equal",
                    "valueText": fact_id
                })
                .with_limit(1)
                .do()
            )
            
            if ("data" in response and "Get" in response["data"] 
                and "Fact" in response["data"]["Get"] 
                and len(response["data"]["Get"]["Fact"]) > 0):
                
                obj = response["data"]["Get"]["Fact"][0]
                return {
                    'fact_id': obj.get('fact_id'),
                    'indicator': obj.get('indicator'), 
                    'district': obj.get('district'),
                    'year': obj.get('year'),
                    'value': obj.get('value'),
                    'unit': obj.get('unit'),
                    'source': obj.get('source'),
                    'span_text': obj.get('span_text'),
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Get by ID failed: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # Get fact count using aggregation
            fact_response = (
                self.client.query
                .aggregate("Fact")
                .with_meta_count()
                .do()
            )
            
            fact_count = 0
            if ("data" in fact_response and "Aggregate" in fact_response["data"] 
                and "Fact" in fact_response["data"]["Aggregate"]
                and len(fact_response["data"]["Aggregate"]["Fact"]) > 0):
                fact_count = fact_response["data"]["Aggregate"]["Fact"][0]["meta"]["count"]
            
            # Get unique indicators
            indicator_response = (
                self.client.query
                .get("Fact", ["indicator"])
                .with_limit(1000)  # Get a sample to count unique values
                .do()
            )
            
            unique_indicators = set()
            if ("data" in indicator_response and "Get" in indicator_response["data"]
                and "Fact" in indicator_response["data"]["Get"]):
                for fact in indicator_response["data"]["Get"]["Fact"]:
                    if fact.get("indicator"):
                        unique_indicators.add(fact["indicator"])
            
            # Get unique districts  
            district_response = (
                self.client.query
                .get("Fact", ["district"])
                .with_limit(1000)
                .do()
            )
            
            unique_districts = set()
            if ("data" in district_response and "Get" in district_response["data"]
                and "Fact" in district_response["data"]["Get"]):
                for fact in district_response["data"]["Get"]["Fact"]:
                    if fact.get("district"):
                        unique_districts.add(fact["district"])
            
            return {
                "total_facts": fact_count,
                "total_documents": 0,  # Will implement later
                "unique_indicators": len(unique_indicators),
                "unique_districts": len(unique_districts),
                "database": "Weaviate"
            }
        
        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            return {}
    
    def close(self):
        """Close connection (v3 client doesn't need explicit close)"""
        pass

# Backward compatibility aliases
PolicyRetriever = WeaviateRetriever