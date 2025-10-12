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
        # Connect to Weaviate
        weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        self.client = weaviate.connect_to_local(
            host=weaviate_url.replace('http://', '').replace('https://', ''),
            port=8080,
            grpc_port=50051
        )
        
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
            fact_collection = self.client.collections.get("Fact")
            
            # Generate query embedding
            query_vector = self.generate_embedding(query)
            
            # Build filter if provided
            filter_obj = None
            if filters:
                filter_conditions = []
                
                if 'indicator' in filters:
                    filter_conditions.append(
                        Filter.by_property("indicator").equal(filters['indicator'])
                    )
                
                if 'district' in filters:
                    filter_conditions.append(
                        Filter.by_property("district").equal(filters['district'])
                    )
                
                if 'year' in filters:
                    filter_conditions.append(
                        Filter.by_property("year").equal(filters['year'])
                    )
                
                if 'category' in filters:
                    filter_conditions.append(
                        Filter.by_property("category").equal(filters['category'])
                    )
                
                # Combine filters with AND
                if filter_conditions:
                    filter_obj = filter_conditions[0]
                    for condition in filter_conditions[1:]:
                        filter_obj = filter_obj & condition
            
            # Hybrid search
            response = fact_collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=limit,
                filters=filter_obj,
                return_metadata=MetadataQuery(distance=True, score=True)
            )
            
            # Format results
            results = []
            for obj in response.objects:
                result = {
                    'fact_id': obj.properties.get('fact_id'),
                    'indicator': obj.properties.get('indicator'),
                    'category': obj.properties.get('category'),
                    'district': obj.properties.get('district'),
                    'year': obj.properties.get('year'),
                    'value': obj.properties.get('value'),
                    'unit': obj.properties.get('unit'),
                    'source': obj.properties.get('source'),
                    'page_ref': obj.properties.get('page_ref'),
                    'confidence': obj.properties.get('confidence'),
                    'span_text': obj.properties.get('span_text'),
                    'pdf_name': obj.properties.get('pdf_name'),
                    'score': obj.metadata.score if obj.metadata else None,
                    'distance': obj.metadata.distance if obj.metadata else None
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results for query: {query}")
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
            fact_collection = self.client.collections.get("Fact")
            
            response = fact_collection.query.fetch_objects(
                filters=Filter.by_property("fact_id").equal(fact_id),
                limit=1
            )
            
            if response.objects:
                obj = response.objects[0]
                return {
                    'fact_id': obj.properties.get('fact_id'),
                    'indicator': obj.properties.get('indicator'),
                    'district': obj.properties.get('district'),
                    'year': obj.properties.get('year'),
                    'value': obj.properties.get('value'),
                    'unit': obj.properties.get('unit'),
                    'source': obj.properties.get('source'),
                    'span_text': obj.properties.get('span_text'),
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Get by ID failed: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            fact_collection = self.client.collections.get("Fact")
            doc_collection = self.client.collections.get("Document")
            
            fact_count = fact_collection.aggregate.over_all(total_count=True).total_count
            doc_count = doc_collection.aggregate.over_all(total_count=True).total_count
            
            # Get unique indicators
            indicator_response = fact_collection.aggregate.over_all(
                group_by="indicator"
            )
            unique_indicators = len(indicator_response.groups) if indicator_response.groups else 0
            
            # Get unique districts
            district_response = fact_collection.aggregate.over_all(
                group_by="district"
            )
            unique_districts = len(district_response.groups) if district_response.groups else 0
            
            return {
                "total_facts": fact_count,
                "total_documents": doc_count,
                "unique_indicators": unique_indicators,
                "unique_districts": unique_districts,
                "database": "Weaviate"
            }
        
        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            return {}
    
    def close(self):
        """Close connection"""
        self.client.close()

# Backward compatibility aliases
PolicyRetriever = WeaviateRetriever