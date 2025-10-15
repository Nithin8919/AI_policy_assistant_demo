#!/usr/bin/env python3
"""
Complete Data Preprocessing and Loading Pipeline for AP Policy Co-Pilot
Properly structures and loads education data for optimal RAG performance
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import weaviate
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess and structure education policy data"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")
    
    def load_normalized_facts(self, file_path: str) -> List[Dict[str, Any]]:
        """Load normalized facts from JSON"""
        logger.info(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            facts = json.load(f)
        logger.info(f"✅ Loaded {len(facts)} facts")
        return facts
    
    def create_searchable_text(self, fact: Dict[str, Any]) -> str:
        """Create rich searchable text from fact"""
        parts = []
        
        # Indicator and value
        indicator = fact.get('indicator', 'Unknown')
        value = fact.get('value', 'N/A')
        unit = fact.get('unit', '')
        parts.append(f"{indicator}: {value} {unit}")
        
        # Location
        district = fact.get('district', 'Unknown')
        parts.append(f"District: {district}")
        
        # Time
        year = fact.get('year')
        if year:
            parts.append(f"Year: {year}")
        
        # Category if exists
        category = fact.get('category')
        if category:
            parts.append(f"Category: {category}")
        
        # Build rich text
        rich_text = f"{indicator} is {value} {unit} in {district} for {year}. "
        rich_text += "This data pertains to education statistics in Andhra Pradesh. "
        rich_text += f"Source: {fact.get('source_document', 'Unknown')}"
        
        return rich_text
    
    def validate_fact(self, fact: Dict[str, Any]) -> bool:
        """Validate that fact has required fields"""
        required = ['fact_id', 'indicator', 'district', 'year', 'value']
        return all(field in fact and fact[field] for field in required)
    
    def preprocess_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess facts for Weaviate"""
        logger.info("Preprocessing facts...")
        processed = []
        skipped = 0
        
        for fact in facts:
            # Validate
            if not self.validate_fact(fact):
                skipped += 1
                continue
            
            # Create searchable text
            searchable_text = self.create_searchable_text(fact)
            
            # Create Weaviate object
            weaviate_obj = {
                'fact_id': str(fact['fact_id']),
                'indicator': str(fact['indicator']),
                'district': str(fact['district']),
                'year': int(fact['year']) if fact['year'] else 0,
                'value': float(fact['value']) if fact['value'] else 0.0,
                'unit': str(fact.get('unit', 'count')),
                'category': str(fact.get('category', 'N/A')) if fact.get('category') else None,
                'source_document': str(fact.get('source_document', 'Unknown')),
                'source_page': int(fact.get('source_page', 1)),
                'confidence': float(fact.get('confidence_score', 0.8)),
                'content': searchable_text,
                'span_text': searchable_text  # For compatibility
            }
            
            processed.append(weaviate_obj)
        
        logger.info(f"✅ Processed {len(processed)} facts")
        if skipped > 0:
            logger.warning(f"⚠️  Skipped {skipped} invalid facts")
        
        return processed
    
    def save_preprocessed(self, facts: List[Dict[str, Any]], output_path: str):
        """Save preprocessed facts"""
        with open(output_path, 'w') as f:
            json.dump(facts, f, indent=2)
        logger.info(f"✅ Saved preprocessed data to {output_path}")


class WeaviateLoader:
    """Load data into Weaviate with proper schema"""
    
    def __init__(self, url: str = "http://localhost:8080"):
        self.client = weaviate.Client(url)
        logger.info("✅ Connected to Weaviate")
    
    def create_schema(self):
        """Create optimized Weaviate schema"""
        logger.info("Creating Weaviate schema...")
        
        # Delete existing class if it exists
        try:
            self.client.schema.delete_class("EducationFact")
            logger.info("   Deleted existing EducationFact class")
        except:
            pass
        
        # Create new schema
        schema = {
            "class": "EducationFact",
            "description": "Structured education statistics for AP",
            "vectorizer": "text2vec-transformers",  # Use built-in vectorizer
            "moduleConfig": {
                "text2vec-transformers": {
                    "poolingStrategy": "masked_mean",
                    "vectorizeClassName": False
                }
            },
            "properties": [
                {
                    "name": "fact_id",
                    "dataType": ["text"],
                    "description": "Unique fact identifier"
                },
                {
                    "name": "indicator",
                    "dataType": ["text"],
                    "description": "Education indicator (e.g., Enrollment, Dropout)"
                },
                {
                    "name": "district",
                    "dataType": ["text"],
                    "description": "AP district name"
                },
                {
                    "name": "year",
                    "dataType": ["int"],
                    "description": "Year of data"
                },
                {
                    "name": "value",
                    "dataType": ["number"],
                    "description": "Numeric value"
                },
                {
                    "name": "unit",
                    "dataType": ["text"],
                    "description": "Unit of measurement"
                },
                {
                    "name": "category",
                    "dataType": ["text"],
                    "description": "Category (SC/ST/OBC/General/etc)"
                },
                {
                    "name": "source_document",
                    "dataType": ["text"],
                    "description": "Source document ID"
                },
                {
                    "name": "source_page",
                    "dataType": ["int"],
                    "description": "Page number in source"
                },
                {
                    "name": "confidence",
                    "dataType": ["number"],
                    "description": "Confidence score (0-1)"
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Rich searchable text"
                },
                {
                    "name": "span_text",
                    "dataType": ["text"],
                    "description": "Display text"
                }
            ]
        }
        
        self.client.schema.create_class(schema)
        logger.info("✅ Schema created successfully")
    
    def load_facts(self, facts: List[Dict[str, Any]], batch_size: int = 100):
        """Load facts into Weaviate with embeddings"""
        logger.info(f"Loading {len(facts)} facts into Weaviate...")
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.client.batch.configure(batch_size=batch_size)
        
        with self.client.batch as batch:
            for i, fact in enumerate(facts):
                try:
                    # Generate embedding
                    vector = embedding_model.encode(fact['content']).tolist()
                    
                    # Add to batch
                    batch.add_data_object(
                        data_object=fact,
                        class_name="EducationFact",
                        vector=vector
                    )
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"   Loaded {i + 1}/{len(facts)} facts...")
                
                except Exception as e:
                    logger.error(f"   Error loading fact {i}: {e}")
                    continue
        
        logger.info(f"✅ Loaded {len(facts)} facts successfully!")
    
    def verify_data(self):
        """Verify loaded data"""
        logger.info("Verifying loaded data...")
        
        result = self.client.query.aggregate("EducationFact").with_meta_count().do()
        count = result['data']['Aggregate']['EducationFact'][0]['meta']['count']
        
        logger.info(f"✅ Total facts in Weaviate: {count}")
        
        # Sample query
        sample = (
            self.client.query
            .get("EducationFact", ["fact_id", "indicator", "district", "year", "value", "content"])
            .with_limit(3)
            .do()
        )
        
        logger.info("\n📊 Sample data:")
        for i, obj in enumerate(sample['data']['Get']['EducationFact'], 1):
            logger.info(f"   {i}. {obj['indicator']} in {obj['district']} ({obj['year']}): {obj['value']}")
        
        return count


def main():
    """Main pipeline execution"""
    logger.info("="*70)
    logger.info("🚀 AP POLICY CO-PILOT DATA PREPROCESSING PIPELINE")
    logger.info("="*70)
    
    # Paths
    input_file = "data/normalized/normalized_facts_enhanced.json"
    output_file = "data/preprocessed_facts.json"
    
    # Step 1: Load and preprocess
    logger.info("\n📊 STEP 1: Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    
    raw_facts = preprocessor.load_normalized_facts(input_file)
    processed_facts = preprocessor.preprocess_facts(raw_facts)
    preprocessor.save_preprocessed(processed_facts, output_file)
    
    # Step 2: Create Weaviate schema
    logger.info("\n🗄️  STEP 2: Creating Weaviate schema...")
    loader = WeaviateLoader()
    loader.create_schema()
    
    # Step 3: Load data
    logger.info("\n📥 STEP 3: Loading data into Weaviate...")
    loader.load_facts(processed_facts)
    
    # Step 4: Verify
    logger.info("\n✅ STEP 4: Verifying loaded data...")
    count = loader.verify_data()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("🎉 PIPELINE COMPLETE!")
    logger.info("="*70)
    logger.info(f"✅ Total facts loaded: {count}")
    logger.info(f"✅ Data saved to: {output_file}")
    logger.info(f"✅ Weaviate class: EducationFact")
    logger.info("\n💡 Next: Restart your API and test queries!")
    logger.info("="*70)


if __name__ == "__main__":
    main()

