#!/usr/bin/env python3
"""
Weaviate Data Loader for AP Policy Co-Pilot
Loads processed facts into Weaviate collections
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from sentence_transformers import SentenceTransformer
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class WeaviateLoader:
    """Load processed facts into Weaviate"""
    
    def __init__(self):
        # Connect to Weaviate
        weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        try:
            self.client = weaviate.connect_to_local(
                host=weaviate_url.replace('http://', '').replace('https://', ''),
                port=8080,
                grpc_port=50051
            )
            logger.info("Connected to Weaviate")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
        
        # Initialize embedding model
        model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedding model: {model_name}")
    
    def setup_collections(self):
        """Setup Weaviate collections with proper schemas"""
        logger.info("Setting up Weaviate collections...")
        
        # Delete existing collections if they exist
        try:
            self.client.collections.delete("Fact")
            self.client.collections.delete("Document") 
            self.client.collections.delete("Entity")
            logger.info("Deleted existing collections")
        except Exception as e:
            logger.debug(f"Collections may not exist: {e}")
        
        # Create Fact collection
        fact_collection = self.client.collections.create(
            name="Fact",
            description="Education policy facts and statistics",
            properties=[
                Property(name="fact_id", data_type=DataType.TEXT),
                Property(name="indicator", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
                Property(name="district", data_type=DataType.TEXT),
                Property(name="year", data_type=DataType.TEXT),
                Property(name="value", data_type=DataType.NUMBER),
                Property(name="unit", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="page_ref", data_type=DataType.INT),
                Property(name="confidence", data_type=DataType.NUMBER),
                Property(name="span_text", data_type=DataType.TEXT),
                Property(name="pdf_name", data_type=DataType.TEXT),
                Property(name="text_id", data_type=DataType.TEXT),
                Property(name="table_id", data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none()  # We'll add vectors manually
        )
        
        # Create Document collection
        doc_collection = self.client.collections.create(
            name="Document",
            description="Source documents and metadata",
            properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="filename", data_type=DataType.TEXT),
                Property(name="source_type", data_type=DataType.TEXT),
                Property(name="file_path", data_type=DataType.TEXT),
                Property(name="year", data_type=DataType.TEXT),
                Property(name="total_pages", data_type=DataType.INT),
                Property(name="extraction_method", data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
        
        # Create Entity collection for named entities
        entity_collection = self.client.collections.create(
            name="Entity",
            description="Named entities (districts, indicators, etc.)",
            properties=[
                Property(name="entity_id", data_type=DataType.TEXT),
                Property(name="entity_name", data_type=DataType.TEXT),
                Property(name="entity_type", data_type=DataType.TEXT),
                Property(name="description", data_type=DataType.TEXT),
                Property(name="created_at", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
        
        logger.info("Collections created successfully")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
    
    def load_facts(self, facts_file: str, batch_size: int = 100):
        """Load facts into Weaviate with embeddings"""
        logger.info(f"Loading facts from {facts_file}")
        
        try:
            with open(facts_file, 'r', encoding='utf-8') as f:
                facts = json.load(f)
        except FileNotFoundError:
            logger.error(f"Facts file not found: {facts_file}")
            return False
        
        fact_collection = self.client.collections.get("Fact")
        total_facts = len(facts)
        
        logger.info(f"Loading {total_facts} facts in batches of {batch_size}")
        
        # Process in batches
        for i in range(0, total_facts, batch_size):
            batch = facts[i:i + batch_size]
            batch_objects = []
            
            for fact in batch:
                try:
                    # Create searchable text from fact
                    search_text = self._create_search_text(fact)
                    
                    # Generate embedding
                    vector = self.generate_embedding(search_text)
                    
                    # Prepare object for Weaviate
                    obj = {
                        "fact_id": fact.get("fact_id", ""),
                        "indicator": fact.get("indicator", ""),
                        "category": fact.get("category", ""),
                        "district": fact.get("district", ""),
                        "year": fact.get("year", ""),
                        "value": float(fact.get("value", 0)),
                        "unit": fact.get("unit", ""),
                        "source": fact.get("source", ""),
                        "page_ref": int(fact.get("page_ref", 0)),
                        "confidence": float(fact.get("confidence", 0.8)),
                        "span_text": fact.get("span_text", "")[:1000],  # Limit length
                        "pdf_name": fact.get("pdf_name", ""),
                        "text_id": fact.get("text_id", ""),
                        "table_id": fact.get("table_id", ""),
                        "created_at": fact.get("created_at", datetime.now().isoformat()),
                    }
                    
                    batch_objects.append({
                        "properties": obj,
                        "vector": vector
                    })
                
                except Exception as e:
                    logger.error(f"Failed to process fact {fact.get('fact_id', 'unknown')}: {e}")
                    continue
            
            # Insert batch
            try:
                response = fact_collection.data.insert_many(batch_objects)
                
                if response.has_errors:
                    for error in response.errors:
                        logger.error(f"Batch insert error: {error}")
                else:
                    logger.info(f"Loaded batch {i//batch_size + 1}/{(total_facts + batch_size - 1)//batch_size} ({len(batch_objects)} facts)")
            
            except Exception as e:
                logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info("Fact loading completed")
        return True
    
    def load_documents(self, extracted_data_file: str):
        """Load document metadata into Weaviate"""
        logger.info(f"Loading document metadata from {extracted_data_file}")
        
        try:
            with open(extracted_data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Extracted data file not found: {extracted_data_file}")
            return False
        
        doc_collection = self.client.collections.get("Document")
        
        for doc_name, items in raw_data.items():
            if not items:
                continue
            
            # Extract document metadata from first item
            first_item = items[0]
            
            doc_obj = {
                "doc_id": str(uuid.uuid4()),
                "filename": doc_name,
                "source_type": first_item.get("source_type", "Unknown"),
                "file_path": first_item.get("file_path", ""),
                "year": first_item.get("year", "Unknown"),
                "total_pages": len(set(item.get("page", 1) for item in items)),
                "extraction_method": ", ".join(set(item.get("extraction_method", "") for item in items)),
                "created_at": datetime.now().isoformat(),
            }
            
            try:
                doc_collection.data.insert(doc_obj)
                logger.debug(f"Loaded document: {doc_name}")
            except Exception as e:
                logger.error(f"Failed to insert document {doc_name}: {e}")
        
        logger.info("Document loading completed")
        return True
    
    def load_entities(self):
        """Load named entities (districts, indicators) into Weaviate"""
        logger.info("Loading named entities")
        
        entity_collection = self.client.collections.get("Entity")
        
        # AP Districts
        ap_districts = [
            'Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Kadapa', 
            'Krishna', 'Kurnool', 'Nellore', 'Prakasam', 'Srikakulam', 
            'Visakhapatnam', 'Vizianagaram', 'West Godavari'
        ]
        
        for district in ap_districts:
            entity_obj = {
                "entity_id": f"DISTRICT_{district.upper().replace(' ', '_')}",
                "entity_name": district,
                "entity_type": "district",
                "description": f"{district} district in Andhra Pradesh",
                "created_at": datetime.now().isoformat(),
            }
            
            try:
                entity_collection.data.insert(entity_obj)
            except Exception as e:
                logger.error(f"Failed to insert district entity {district}: {e}")
        
        # Education Indicators
        indicators = [
            ("GER", "Gross Enrolment Ratio", "Ratio of total enrollment to population of official school age"),
            ("NER", "Net Enrolment Ratio", "Ratio of children of official school age enrolled to population of same age"),
            ("PTR", "Pupil Teacher Ratio", "Number of pupils per teacher"),
            ("Dropout_Rate", "Dropout Rate", "Percentage of students who leave school before completion"),
            ("Enrollment", "Student Enrollment", "Total number of students enrolled"),
            ("Schools", "Number of Schools", "Total count of educational institutions"),
            ("Teachers", "Number of Teachers", "Total count of teaching staff"),
            ("Budget", "Budget Allocation", "Financial resources allocated for education"),
        ]
        
        for indicator_code, name, description in indicators:
            entity_obj = {
                "entity_id": f"INDICATOR_{indicator_code}",
                "entity_name": name,
                "entity_type": "indicator",
                "description": description,
                "created_at": datetime.now().isoformat(),
            }
            
            try:
                entity_collection.data.insert(entity_obj)
            except Exception as e:
                logger.error(f"Failed to insert indicator entity {indicator_code}: {e}")
        
        logger.info("Entity loading completed")
        return True
    
    def _create_search_text(self, fact: Dict[str, Any]) -> str:
        """Create searchable text from fact"""
        components = [
            fact.get("indicator", ""),
            fact.get("district", ""),
            fact.get("year", ""),
            str(fact.get("value", "")),
            fact.get("unit", ""),
            fact.get("source", ""),
            fact.get("span_text", "")[:200]  # First 200 chars
        ]
        
        return " ".join(str(c) for c in components if c)
    
    def verify_loading(self) -> Dict[str, Any]:
        """Verify that data was loaded correctly"""
        logger.info("Verifying data loading...")
        
        try:
            fact_collection = self.client.collections.get("Fact")
            doc_collection = self.client.collections.get("Document")
            entity_collection = self.client.collections.get("Entity")
            
            fact_count = fact_collection.aggregate.over_all(total_count=True).total_count
            doc_count = doc_collection.aggregate.over_all(total_count=True).total_count
            entity_count = entity_collection.aggregate.over_all(total_count=True).total_count
            
            # Test a simple search
            response = fact_collection.query.fetch_objects(limit=3)
            sample_facts = [obj.properties for obj in response.objects]
            
            verification_result = {
                "fact_count": fact_count,
                "document_count": doc_count,
                "entity_count": entity_count,
                "sample_facts": sample_facts,
                "status": "success" if fact_count > 0 else "warning",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Verification complete: {fact_count} facts, {doc_count} docs, {entity_count} entities")
            return verification_result
        
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def close(self):
        """Close Weaviate connection"""
        self.client.close()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load processed data into Weaviate')
    parser.add_argument('--facts-file', default='data/processed/processed_facts.json',
                       help='Processed facts JSON file')
    parser.add_argument('--extracted-file', default='data/extracted/all_extracted_data.json',
                       help='Original extracted data file')
    parser.add_argument('--setup-collections', action='store_true',
                       help='Setup/reset collections before loading')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for loading facts')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    loader = WeaviateLoader()
    
    try:
        if args.setup_collections:
            loader.setup_collections()
        
        # Load facts
        if os.path.exists(args.facts_file):
            loader.load_facts(args.facts_file, args.batch_size)
        else:
            logger.error(f"Facts file not found: {args.facts_file}")
        
        # Load documents
        if os.path.exists(args.extracted_file):
            loader.load_documents(args.extracted_file)
        else:
            logger.warning(f"Extracted data file not found: {args.extracted_file}")
        
        # Load entities
        loader.load_entities()
        
        # Verify loading
        result = loader.verify_loading()
        
        print(f"\n✅ Data loading completed!")
        print(f"📊 Facts loaded: {result.get('fact_count', 0):,}")
        print(f"📄 Documents loaded: {result.get('document_count', 0):,}")
        print(f"🏷️ Entities loaded: {result.get('entity_count', 0):,}")
        print(f"Status: {result.get('status', 'unknown')}")
    
    finally:
        loader.close()

if __name__ == "__main__":
    main()