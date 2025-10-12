#!/usr/bin/env python3
"""
Stage 3: Fact Table Builder (Weaviate)
Loads normalized facts into Weaviate vector database
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import uuid


import weaviate
from weaviate.classes.data import DataObject
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class WeaviateFactLoader:
    """Load facts into Weaviate"""
    
    def __init__(self, output_dir: str = "data/weaviate"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Weaviate connection
        weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        self.client = weaviate.connect_to_local(
            host=weaviate_url.replace('http://', '').replace('https://', ''),
            port=8080,
            grpc_port=50051
        )
        
        # Embedding model
        model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        
        logger.info(f"Initialized Weaviate loader with {model_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not text or text.strip() == "":
            # Return zero vector for empty text
            return [0.0] * 384
        
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def load_facts(self, normalized_facts: List[Dict[str, Any]]) -> bool:
        """Load facts into Weaviate"""
        logger.info(f"Loading {len(normalized_facts)} facts into Weaviate...")
        
        try:
            fact_collection = self.client.collections.get("Fact")
            
            # Batch insert
            with fact_collection.batch.dynamic() as batch:
                for idx, fact in enumerate(normalized_facts):
                    
                    if idx % 100 == 0:
                        logger.info(f"Processing fact {idx}/{len(normalized_facts)}")
                    
                    # Generate embedding from span_text
                    span_text = fact.get('span_text', '')
                    embedding = self.generate_embedding(span_text)
                    
                    # Create data object
                    data_object = {
                        "fact_id": fact.get('fact_id', str(uuid.uuid4())),
                        "indicator": fact.get('indicator', 'Unknown'),
                        "category": fact.get('category', 'Unknown'),
                        "district": fact.get('district', 'Unknown'),
                        "year": fact.get('year', 'Unknown'),
                        "value": float(fact.get('value', 0.0)) if fact.get('value') is not None else 0.0,
                        "unit": fact.get('unit', ''),
                        "source": fact.get('source', ''),
                        "page_ref": int(fact.get('page_ref', 0)) if fact.get('page_ref') else 0,
                        "confidence": float(fact.get('confidence', 0.0)),
                        "table_id": fact.get('table_id', ''),
                        "pdf_name": fact.get('pdf_name', ''),
                        "span_text": span_text,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Add to batch with vector
                    batch.add_object(
                        properties=data_object,
                        vector=embedding
                    )
            
            logger.info(f"✅ Successfully loaded {len(normalized_facts)} facts")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load facts: {e}")
            return False
    
    def load_documents(self, extracted_data: List[Dict[str, Any]]) -> bool:
        """Load document metadata into Weaviate"""
        logger.info(f"Loading {len(extracted_data)} documents into Weaviate...")
        
        try:
            doc_collection = self.client.collections.get("Document")
            
            # Extract unique documents
            documents = {}
            for item in extracted_data:
                doc_id = item.get('doc_id')
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "doc_id": doc_id,
                        "filename": item.get('filename', ''),
                        "source_type": item.get('source_type', ''),
                        "year": item.get('year', ''),
                        "total_pages": int(item.get('page', 1)),
                        "extraction_method": item.get('extraction_method', ''),
                        "checksum": item.get('checksum', ''),
                        "file_path": item.get('file_path', ''),
                        "created_at": datetime.now().isoformat()
                    }
            
            # Batch insert
            with doc_collection.batch.dynamic() as batch:
                for doc_data in documents.values():
                    batch.add_object(properties=doc_data)
            
            logger.info(f"✅ Successfully loaded {len(documents)} documents")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return False
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary of loaded data"""
        try:
            fact_collection = self.client.collections.get("Fact")
            doc_collection = self.client.collections.get("Document")
            
            fact_count = fact_collection.aggregate.over_all(total_count=True).total_count
            doc_count = doc_collection.aggregate.over_all(total_count=True).total_count
            
            summary = {
                "total_facts": fact_count,
                "total_documents": doc_count,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save summary
            summary_file = self.output_dir / "weaviate_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            return summary
        
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {}
    
    def close(self):
        """Close Weaviate connection"""
        self.client.close()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load facts into Weaviate')
    parser.add_argument('--normalized-file', default='data/normalized/normalized_facts.json')
    parser.add_argument('--extracted-file', default='data/extracted/all_extracted_data.json')
    parser.add_argument('--output-dir', default='data/weaviate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    try:
        with open(args.normalized_file, 'r') as f:
            normalized_facts = json.load(f)
        logger.info(f"Loaded {len(normalized_facts)} normalized facts")
    except FileNotFoundError:
        logger.error(f"Normalized facts file not found: {args.normalized_file}")
        return
    
    try:
        with open(args.extracted_file, 'r') as f:
            extracted_data = json.load(f)
        logger.info(f"Loaded {len(extracted_data)} extracted items")
    except FileNotFoundError:
        logger.error(f"Extracted data file not found: {args.extracted_file}")
        return
    
    # Initialize loader
    loader = WeaviateFactLoader(output_dir=args.output_dir)
    
    try:
        # Load facts and documents
        success = True
        success &= loader.load_facts(normalized_facts)
        success &= loader.load_documents(extracted_data)
        
        if success:
            # Generate summary
            summary = loader.generate_summary()
            
            print(f"\n✅ Weaviate Loading Complete:")
            print(f"   Total facts: {summary.get('total_facts', 0)}")
            print(f"   Total documents: {summary.get('total_documents', 0)}")
            print(f"   Output directory: {args.output_dir}")
        else:
            logger.error("Weaviate loading failed")
    
    finally:
        loader.close()

if __name__ == "__main__":
    main()