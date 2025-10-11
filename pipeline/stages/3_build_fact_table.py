#!/usr/bin/env python3
"""
Stage 3: Fact Table Builder
Creates PostgreSQL bridge table with pgvector for hybrid retrieval
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib

# Vector embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class FactTableBuilder:
    """Production-ready fact table builder with vector embeddings"""
    
    def __init__(self, output_dir: str = "data/bridge_table"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'ap_education_policy',
            'user': 'postgres',
            'password': 'password'
        }
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedding_model = None
            logger.warning("SentenceTransformers not available. Embeddings will be disabled.")
        
        # Table schemas
        self.fact_table_schema = self._get_fact_table_schema()
        self.document_table_schema = self._get_document_table_schema()
        self.entity_table_schema = self._get_entity_table_schema()
    
    def _get_fact_table_schema(self) -> str:
        """Get fact table schema with pgvector support"""
        return """
        CREATE TABLE IF NOT EXISTS facts (
            fact_id VARCHAR(50) PRIMARY KEY,
            indicator VARCHAR(100) NOT NULL,
            category VARCHAR(50),
            district VARCHAR(100),
            year VARCHAR(20),
            value DECIMAL(15,6),
            unit VARCHAR(20),
            source VARCHAR(50),
            page_ref INTEGER,
            confidence DECIMAL(3,2),
            table_id VARCHAR(50),
            pdf_name VARCHAR(200),
            span_text TEXT,
            embedding VECTOR(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_facts_indicator ON facts(indicator);
        CREATE INDEX IF NOT EXISTS idx_facts_district ON facts(district);
        CREATE INDEX IF NOT EXISTS idx_facts_year ON facts(year);
        CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source);
        CREATE INDEX IF NOT EXISTS idx_facts_embedding ON facts USING ivfflat (embedding vector_cosine_ops);
        """
    
    def _get_document_table_schema(self) -> str:
        """Get document table schema"""
        return """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id VARCHAR(50) PRIMARY KEY,
            filename VARCHAR(200) NOT NULL,
            source_type VARCHAR(50),
            year VARCHAR(20),
            total_pages INTEGER,
            extraction_method VARCHAR(50),
            checksum VARCHAR(64),
            file_path VARCHAR(500),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_type);
        CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year);
        """
    
    def _get_entity_table_schema(self) -> str:
        """Get entity table schema"""
        return """
        CREATE TABLE IF NOT EXISTS entities (
            entity_id VARCHAR(50) PRIMARY KEY,
            entity_type VARCHAR(50) NOT NULL,
            entity_name VARCHAR(200) NOT NULL,
            canonical_name VARCHAR(200),
            aliases TEXT[],
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(entity_name);
        """
    
    def setup_database(self) -> bool:
        """Setup PostgreSQL database with pgvector extension"""
        try:
            # Connect to PostgreSQL
            conn = psycopg2.connect(**self.db_config)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create tables
            cursor.execute(self.fact_table_schema)
            cursor.execute(self.document_table_schema)
            cursor.execute(self.entity_table_schema)
            
            # Create views for common queries
            self._create_views(cursor)
            
            cursor.close()
            conn.close()
            
            logger.info("Database setup completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def _create_views(self, cursor):
        """Create useful views for querying"""
        views = [
            """
            CREATE OR REPLACE VIEW fact_summary AS
            SELECT 
                indicator,
                district,
                year,
                COUNT(*) as fact_count,
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                source
            FROM facts
            GROUP BY indicator, district, year, source;
            """,
            
            """
            CREATE OR REPLACE VIEW district_stats AS
            SELECT 
                district,
                COUNT(DISTINCT indicator) as indicator_count,
                COUNT(*) as total_facts,
                COUNT(DISTINCT year) as year_count
            FROM facts
            GROUP BY district;
            """,
            
            """
            CREATE OR REPLACE VIEW indicator_trends AS
            SELECT 
                indicator,
                year,
                COUNT(DISTINCT district) as district_count,
                AVG(value) as avg_value,
                source
            FROM facts
            GROUP BY indicator, year, source
            ORDER BY indicator, year;
            """
        ]
        
        for view_sql in views:
            try:
                cursor.execute(view_sql)
            except Exception as e:
                logger.warning(f"Failed to create view: {e}")
    
    def build_fact_table(self, normalized_facts: List[Dict[str, Any]]) -> bool:
        """
        Build fact table from normalized data
        
        Args:
            normalized_facts: List of normalized facts from Stage 2
            
        Returns:
            Success status
        """
        logger.info(f"Building fact table with {len(normalized_facts)} facts")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("TRUNCATE TABLE facts CASCADE;")
            
            # Insert facts
            inserted_count = 0
            for fact in normalized_facts:
                try:
                    # Generate span text for embedding
                    span_text = self._generate_span_text(fact)
                    
                    # Generate embedding
                    embedding = self._generate_embedding(span_text)
                    
                    # Insert fact
                    insert_sql = """
                    INSERT INTO facts (
                        fact_id, indicator, category, district, year, value, unit,
                        source, page_ref, confidence, table_id, pdf_name, span_text, embedding
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    );
                    """
                    
                    cursor.execute(insert_sql, (
                        fact['fact_id'],
                        fact['indicator'],
                        fact.get('category', 'total'),
                        fact['district'],
                        fact['year'],
                        fact['value'],
                        fact.get('unit', 'unknown'),
                        fact['source'],
                        fact.get('page_ref', 0),
                        fact.get('confidence', 0.8),
                        fact.get('table_id', ''),
                        fact.get('pdf_name', ''),
                        span_text,
                        embedding
                    ))
                    
                    inserted_count += 1
                
                except Exception as e:
                    logger.error(f"Failed to insert fact {fact.get('fact_id', 'unknown')}: {e}")
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Successfully inserted {inserted_count} facts")
            return True
        
        except Exception as e:
            logger.error(f"Fact table build failed: {e}")
            return False
    
    def _generate_span_text(self, fact: Dict[str, Any]) -> str:
        """Generate span text for embedding"""
        # Create descriptive text for the fact
        span_parts = [
            f"{fact['indicator']}",
            f"in {fact['district']}",
            f"for {fact['year']}",
            f"is {fact['value']}",
            f"{fact.get('unit', '')}"
        ]
        
        return " ".join(span_parts).strip()
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        if not self.embedding_model:
            return None
        
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def build_document_table(self, extracted_data: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Build document table from extracted data"""
        logger.info("Building document table")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("TRUNCATE TABLE documents CASCADE;")
            
            # Insert documents
            for pdf_name, items in extracted_data.items():
                try:
                    # Get document metadata from first item
                    first_item = items[0] if items else {}
                    
                    # Calculate checksum
                    checksum = self._calculate_document_checksum(items)
                    
                    # Count pages
                    pages = max(item.get('page', 0) for item in items) if items else 0
                    
                    insert_sql = """
                    INSERT INTO documents (
                        doc_id, filename, source_type, year, total_pages,
                        extraction_method, checksum, file_path
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s
                    );
                    """
                    
                    cursor.execute(insert_sql, (
                        Path(pdf_name).stem,
                        pdf_name,
                        first_item.get('source_type', 'Unknown'),
                        first_item.get('year', 'Unknown'),
                        pages,
                        first_item.get('extraction_method', 'Unknown'),
                        checksum,
                        f"data/preprocessed/documents/{pdf_name}"
                    ))
                
                except Exception as e:
                    logger.error(f"Failed to insert document {pdf_name}: {e}")
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Document table built successfully")
            return True
        
        except Exception as e:
            logger.error(f"Document table build failed: {e}")
            return False
    
    def _calculate_document_checksum(self, items: List[Dict[str, Any]]) -> str:
        """Calculate checksum for document"""
        # Create a stable string representation
        data_str = json.dumps(items, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def build_entity_table(self, normalized_facts: List[Dict[str, Any]]) -> bool:
        """Build entity table from normalized facts"""
        logger.info("Building entity table")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("TRUNCATE TABLE entities CASCADE;")
            
            # Extract unique entities
            entities = self._extract_entities(normalized_facts)
            
            # Insert entities
            for entity in entities:
                try:
                    insert_sql = """
                    INSERT INTO entities (
                        entity_id, entity_type, entity_name, canonical_name,
                        aliases, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s
                    );
                    """
                    
                    cursor.execute(insert_sql, (
                        entity['entity_id'],
                        entity['entity_type'],
                        entity['entity_name'],
                        entity['canonical_name'],
                        entity['aliases'],
                        json.dumps(entity['metadata'])
                    ))
                
                except Exception as e:
                    logger.error(f"Failed to insert entity {entity.get('entity_id', 'unknown')}: {e}")
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Entity table built successfully")
            return True
        
        except Exception as e:
            logger.error(f"Entity table build failed: {e}")
            return False
    
    def _extract_entities(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique entities from facts"""
        entities = []
        entity_id_counter = 1
        
        # Extract districts
        districts = set(fact['district'] for fact in facts if fact['district'] != 'Unknown')
        for district in districts:
            entities.append({
                'entity_id': f"ENT_{entity_id_counter:06d}",
                'entity_type': 'District',
                'entity_name': district,
                'canonical_name': district,
                'aliases': [district.lower()],
                'metadata': {'type': 'geographic', 'state': 'Andhra Pradesh'}
            })
            entity_id_counter += 1
        
        # Extract indicators
        indicators = set(fact['indicator'] for fact in facts if fact['indicator'] != 'Unknown')
        for indicator in indicators:
            entities.append({
                'entity_id': f"ENT_{entity_id_counter:06d}",
                'entity_type': 'Indicator',
                'entity_name': indicator,
                'canonical_name': indicator,
                'aliases': [indicator.lower()],
                'metadata': {'type': 'statistical', 'domain': 'education'}
            })
            entity_id_counter += 1
        
        # Extract sources
        sources = set(fact['source'] for fact in facts if fact['source'] != 'Unknown')
        for source in sources:
            entities.append({
                'entity_id': f"ENT_{entity_id_counter:06d}",
                'entity_type': 'Source',
                'entity_name': source,
                'canonical_name': source,
                'aliases': [source.lower()],
                'metadata': {'type': 'institutional', 'domain': 'education'}
            })
            entity_id_counter += 1
        
        return entities
    
    def generate_bridge_table_summary(self) -> Dict[str, Any]:
        """Generate summary of bridge table contents"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get fact count
            cursor.execute("SELECT COUNT(*) as count FROM facts;")
            fact_count = cursor.fetchone()['count']
            
            # Get document count
            cursor.execute("SELECT COUNT(*) as count FROM documents;")
            doc_count = cursor.fetchone()['count']
            
            # Get entity count
            cursor.execute("SELECT COUNT(*) as count FROM entities;")
            entity_count = cursor.fetchone()['count']
            
            # Get indicator distribution
            cursor.execute("""
                SELECT indicator, COUNT(*) as count 
                FROM facts 
                GROUP BY indicator 
                ORDER BY count DESC 
                LIMIT 10;
            """)
            top_indicators = cursor.fetchall()
            
            # Get district distribution
            cursor.execute("""
                SELECT district, COUNT(*) as count 
                FROM facts 
                GROUP BY district 
                ORDER BY count DESC 
                LIMIT 10;
            """)
            top_districts = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                'total_facts': fact_count,
                'total_documents': doc_count,
                'total_entities': entity_count,
                'top_indicators': [dict(row) for row in top_indicators],
                'top_districts': [dict(row) for row in top_districts]
            }
        
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {}
    
    def export_bridge_table(self) -> bool:
        """Export bridge table to files"""
        try:
            conn = psycopg2.connect(**self.db_config)
            
            # Export facts
            facts_df = pd.read_sql("SELECT * FROM facts;", conn)
            facts_file = self.output_dir / "facts.csv"
            facts_df.to_csv(facts_file, index=False)
            
            # Export documents
            docs_df = pd.read_sql("SELECT * FROM documents;", conn)
            docs_file = self.output_dir / "documents.csv"
            docs_df.to_csv(docs_file, index=False)
            
            # Export entities
            entities_df = pd.read_sql("SELECT * FROM entities;", conn)
            entities_file = self.output_dir / "entities.csv"
            entities_df.to_csv(entities_file, index=False)
            
            conn.close()
            
            logger.info(f"Bridge table exported to {self.output_dir}")
            return True
        
        except Exception as e:
            logger.error(f"Bridge table export failed: {e}")
            return False

def main():
    """Main function to run fact table builder"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build fact table')
    parser.add_argument('--normalized-file', default='data/normalized/normalized_facts.json',
                       help='Input file with normalized facts')
    parser.add_argument('--extracted-file', default='data/extracted/all_extracted_data.json',
                       help='Input file with extracted data')
    parser.add_argument('--output-dir', default='data/bridge_table',
                       help='Output directory for bridge table')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load normalized facts
    try:
        with open(args.normalized_file, 'r', encoding='utf-8') as f:
            normalized_facts = json.load(f)
    except FileNotFoundError:
        logger.error(f"Normalized facts file not found: {args.normalized_file}")
        return
    
    # Load extracted data
    try:
        with open(args.extracted_file, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Extracted data file not found: {args.extracted_file}")
        return
    
    # Initialize builder
    builder = FactTableBuilder(output_dir=args.output_dir)
    
    # Setup database
    if not builder.setup_database():
        logger.error("Database setup failed")
        return
    
    # Build tables
    success = True
    success &= builder.build_fact_table(normalized_facts)
    success &= builder.build_document_table(extracted_data)
    success &= builder.build_entity_table(normalized_facts)
    
    if success:
        # Generate summary
        summary = builder.generate_bridge_table_summary()
        
        # Export data
        builder.export_bridge_table()
        
        # Print summary
        print(f"\nFact Table Summary:")
        print(f"Total facts: {summary.get('total_facts', 0)}")
        print(f"Total documents: {summary.get('total_documents', 0)}")
        print(f"Total entities: {summary.get('total_entities', 0)}")
        print(f"Output directory: {args.output_dir}")
    else:
        logger.error("Fact table build failed")

if __name__ == "__main__":
    main()
