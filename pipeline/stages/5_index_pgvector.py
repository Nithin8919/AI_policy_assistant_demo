#!/usr/bin/env python3
"""
Stage 5: pgvector Index Builder
Creates vector embeddings and indexes for semantic search
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

# Vector embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class PgVectorIndexBuilder:
    """Production-ready pgvector index builder for semantic search"""
    
    def __init__(self, output_dir: str = "data/embeddings"):
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
            self.embedding_dimension = 384
        else:
            self.embedding_model = None
            self.embedding_dimension = 384
            logger.warning("SentenceTransformers not available. Using dummy embeddings.")
        
        # Query templates for semantic search
        self.query_templates = self._build_query_templates()
    
    def _build_query_templates(self) -> Dict[str, str]:
        """Build query templates for different types of semantic searches"""
        return {
            'semantic_search': """
                SELECT 
                    f.fact_id,
                    f.indicator,
                    f.district,
                    f.year,
                    f.value,
                    f.unit,
                    f.source,
                    f.span_text,
                    f.confidence,
                    1 - (f.embedding <=> %s) as similarity_score
                FROM facts f
                WHERE f.embedding IS NOT NULL
                ORDER BY f.embedding <=> %s
                LIMIT %s
            """,
            
            'hybrid_search': """
                SELECT 
                    f.fact_id,
                    f.indicator,
                    f.district,
                    f.year,
                    f.value,
                    f.unit,
                    f.source,
                    f.span_text,
                    f.confidence,
                    (1 - (f.embedding <=> %s)) * %s + 
                    CASE 
                        WHEN f.indicator ILIKE %s THEN 0.3
                        WHEN f.district ILIKE %s THEN 0.2
                        WHEN f.year ILIKE %s THEN 0.1
                        ELSE 0
                    END as combined_score
                FROM facts f
                WHERE f.embedding IS NOT NULL
                ORDER BY combined_score DESC
                LIMIT %s
            """,
            
            'indicator_search': """
                SELECT 
                    f.fact_id,
                    f.indicator,
                    f.district,
                    f.year,
                    f.value,
                    f.unit,
                    f.source,
                    f.span_text,
                    f.confidence,
                    1 - (f.embedding <=> %s) as similarity_score
                FROM facts f
                WHERE f.embedding IS NOT NULL
                AND f.indicator = %s
                ORDER BY f.embedding <=> %s
                LIMIT %s
            """,
            
            'district_search': """
                SELECT 
                    f.fact_id,
                    f.indicator,
                    f.district,
                    f.year,
                    f.value,
                    f.unit,
                    f.source,
                    f.span_text,
                    f.confidence,
                    1 - (f.embedding <=> %s) as similarity_score
                FROM facts f
                WHERE f.embedding IS NOT NULL
                AND f.district = %s
                ORDER BY f.embedding <=> %s
                LIMIT %s
            """,
            
            'temporal_search': """
                SELECT 
                    f.fact_id,
                    f.indicator,
                    f.district,
                    f.year,
                    f.value,
                    f.unit,
                    f.source,
                    f.span_text,
                    f.confidence,
                    1 - (f.embedding <=> %s) as similarity_score
                FROM facts f
                WHERE f.embedding IS NOT NULL
                AND f.year BETWEEN %s AND %s
                ORDER BY f.embedding <=> %s
                LIMIT %s
            """
        }
    
    def generate_embeddings(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for facts
        
        Args:
            facts: List of facts from bridge table
            
        Returns:
            List of facts with embeddings
        """
        logger.info(f"Generating embeddings for {len(facts)} facts")
        
        facts_with_embeddings = []
        
        for fact in facts:
            try:
                # Generate span text for embedding
                span_text = self._generate_span_text(fact)
                
                # Generate embedding
                embedding = self._generate_embedding(span_text)
                
                if embedding:
                    fact['span_text'] = span_text
                    fact['embedding'] = embedding
                    facts_with_embeddings.append(fact)
                else:
                    logger.warning(f"Failed to generate embedding for fact {fact.get('fact_id', 'unknown')}")
            
            except Exception as e:
                logger.error(f"Embedding generation failed for fact {fact.get('fact_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Generated embeddings for {len(facts_with_embeddings)} facts")
        return facts_with_embeddings
    
    def _generate_span_text(self, fact: Dict[str, Any]) -> str:
        """Generate descriptive span text for embedding"""
        # Create comprehensive description
        span_parts = [
            f"The {fact['indicator']}",
            f"in {fact['district']} district",
            f"for the year {fact['year']}",
            f"is {fact['value']}",
            f"{fact.get('unit', '')}",
            f"as reported by {fact['source']}"
        ]
        
        # Add category information
        if fact.get('category') and fact['category'] != 'total':
            span_parts.append(f"for {fact['category']} category")
        
        return " ".join(span_parts).strip()
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        if not self.embedding_model:
            # Return dummy embedding for testing
            return [0.1] * self.embedding_dimension
        
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def update_fact_embeddings(self) -> bool:
        """Update embeddings for all facts in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get all facts without embeddings
            cursor.execute("""
                SELECT fact_id, indicator, category, district, year, value, unit, source
                FROM facts 
                WHERE embedding IS NULL
            """)
            
            facts = cursor.fetchall()
            logger.info(f"Found {len(facts)} facts without embeddings")
            
            # Generate embeddings
            facts_with_embeddings = self.generate_embeddings([dict(fact) for fact in facts])
            
            # Update database
            updated_count = 0
            for fact in facts_with_embeddings:
                try:
                    update_sql = """
                    UPDATE facts 
                    SET span_text = %s, embedding = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE fact_id = %s
                    """
                    
                    cursor.execute(update_sql, (
                        fact['span_text'],
                        fact['embedding'],
                        fact['fact_id']
                    ))
                    
                    updated_count += 1
                
                except Exception as e:
                    logger.error(f"Failed to update fact {fact.get('fact_id', 'unknown')}: {e}")
                    continue
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Updated embeddings for {updated_count} facts")
            return True
        
        except Exception as e:
            logger.error(f"Embedding update failed: {e}")
            return False
    
    def create_vector_indexes(self) -> bool:
        """Create vector indexes for efficient search"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create vector index
            index_queries = [
                """
                CREATE INDEX IF NOT EXISTS idx_facts_embedding_cosine 
                ON facts USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100)
                """,
                
                """
                CREATE INDEX IF NOT EXISTS idx_facts_embedding_l2 
                ON facts USING ivfflat (embedding vector_l2_ops) 
                WITH (lists = 100)
                """,
                
                """
                CREATE INDEX IF NOT EXISTS idx_facts_embedding_ip 
                ON facts USING ivfflat (embedding vector_ip_ops) 
                WITH (lists = 100)
                """
            ]
            
            for query in index_queries:
                try:
                    cursor.execute(query)
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Vector indexes created successfully")
            return True
        
        except Exception as e:
            logger.error(f"Vector index creation failed: {e}")
            return False
    
    def semantic_search(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search
        
        Args:
            query_text: Search query text
            limit: Maximum number of results
            
        Returns:
            List of matching facts
        """
        if not self.embedding_model:
            logger.error("Embedding model not available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query_text)
            if not query_embedding:
                return []
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Execute semantic search
            cursor.execute(self.query_templates['semantic_search'], 
                         (query_embedding, query_embedding, limit))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def hybrid_search(self, query_text: str, indicator: str = None, 
                     district: str = None, year: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching
        
        Args:
            query_text: Search query text
            indicator: Filter by indicator
            district: Filter by district
            year: Filter by year
            limit: Maximum number of results
            
        Returns:
            List of matching facts
        """
        if not self.embedding_model:
            logger.error("Embedding model not available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query_text)
            if not query_embedding:
                return []
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Execute hybrid search
            cursor.execute(self.query_templates['hybrid_search'], (
                query_embedding, 0.7,  # 70% semantic, 30% keyword
                f"%{indicator}%" if indicator else "",
                f"%{district}%" if district else "",
                f"%{year}%" if year else "",
                limit
            ))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def get_similar_facts(self, fact_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find facts similar to a given fact
        
        Args:
            fact_id: ID of the reference fact
            limit: Maximum number of similar facts
            
        Returns:
            List of similar facts
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get reference fact embedding
            cursor.execute("SELECT embedding FROM facts WHERE fact_id = %s", (fact_id,))
            result = cursor.fetchone()
            
            if not result or not result['embedding']:
                logger.warning(f"No embedding found for fact {fact_id}")
                return []
            
            reference_embedding = result['embedding']
            
            # Find similar facts
            cursor.execute("""
                SELECT 
                    f.fact_id,
                    f.indicator,
                    f.district,
                    f.year,
                    f.value,
                    f.unit,
                    f.source,
                    f.span_text,
                    f.confidence,
                    1 - (f.embedding <=> %s) as similarity_score
                FROM facts f
                WHERE f.fact_id != %s AND f.embedding IS NOT NULL
                ORDER BY f.embedding <=> %s
                LIMIT %s
            """, (reference_embedding, fact_id, reference_embedding, limit))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(result) for result in results]
        
        except Exception as e:
            logger.error(f"Similar facts search failed: {e}")
            return []
    
    def generate_embedding_summary(self) -> Dict[str, Any]:
        """Generate summary of embedding coverage"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get embedding statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_facts,
                    COUNT(embedding) as facts_with_embeddings,
                    COUNT(*) - COUNT(embedding) as facts_without_embeddings
                FROM facts
            """)
            
            stats = cursor.fetchone()
            
            # Get embedding quality metrics
            cursor.execute("""
                SELECT 
                    AVG(confidence) as avg_confidence,
                    MIN(confidence) as min_confidence,
                    MAX(confidence) as max_confidence
                FROM facts
                WHERE embedding IS NOT NULL
            """)
            
            quality_stats = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return {
                'total_facts': stats['total_facts'],
                'facts_with_embeddings': stats['facts_with_embeddings'],
                'facts_without_embeddings': stats['facts_without_embeddings'],
                'embedding_coverage': stats['facts_with_embeddings'] / stats['total_facts'] if stats['total_facts'] > 0 else 0,
                'avg_confidence': quality_stats['avg_confidence'],
                'min_confidence': quality_stats['min_confidence'],
                'max_confidence': quality_stats['max_confidence']
            }
        
        except Exception as e:
            logger.error(f"Embedding summary generation failed: {e}")
            return {}
    
    def export_embeddings(self) -> bool:
        """Export embeddings to files"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Export facts with embeddings
            cursor.execute("""
                SELECT fact_id, indicator, district, year, value, unit, 
                       source, span_text, embedding, confidence
                FROM facts
                WHERE embedding IS NOT NULL
            """)
            
            results = cursor.fetchall()
            
            # Convert to list of dicts
            facts_data = []
            for result in results:
                fact_dict = dict(result)
                # Convert numpy array to list if needed
                if hasattr(fact_dict['embedding'], 'tolist'):
                    fact_dict['embedding'] = fact_dict['embedding'].tolist()
                facts_data.append(fact_dict)
            
            # Save to JSON
            embeddings_file = self.output_dir / "fact_embeddings.json"
            with open(embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(facts_data, f, indent=2, ensure_ascii=False)
            
            # Save to CSV (without embeddings)
            csv_data = []
            for fact in facts_data:
                csv_fact = {k: v for k, v in fact.items() if k != 'embedding'}
                csv_data.append(csv_fact)
            
            df = pd.DataFrame(csv_data)
            csv_file = self.output_dir / "fact_embeddings.csv"
            df.to_csv(csv_file, index=False)
            
            cursor.close()
            conn.close()
            
            logger.info(f"Embeddings exported to {self.output_dir}")
            return True
        
        except Exception as e:
            logger.error(f"Embedding export failed: {e}")
            return False

def main():
    """Main function to run pgvector index builder"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build pgvector indexes')
    parser.add_argument('--normalized-file', default='data/normalized/normalized_facts.json',
                       help='Input file with normalized facts')
    parser.add_argument('--output-dir', default='data/embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--update-only', action='store_true',
                       help='Only update existing embeddings')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize builder
    builder = PgVectorIndexBuilder(output_dir=args.output_dir)
    
    if args.update_only:
        # Update existing embeddings
        success = builder.update_fact_embeddings()
        if success:
            builder.create_vector_indexes()
            builder.export_embeddings()
    else:
        # Load normalized facts and generate embeddings
        try:
            with open(args.normalized_file, 'r', encoding='utf-8') as f:
                normalized_facts = json.load(f)
        except FileNotFoundError:
            logger.error(f"Normalized facts file not found: {args.normalized_file}")
            return
        
        # Generate embeddings
        facts_with_embeddings = builder.generate_embeddings(normalized_facts)
        
        # Save embeddings
        embeddings_file = args.output_dir / "generated_embeddings.json"
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(facts_with_embeddings, f, indent=2, ensure_ascii=False)
        
        # Update database
        builder.update_fact_embeddings()
        builder.create_vector_indexes()
        builder.export_embeddings()
    
    # Generate summary
    summary = builder.generate_embedding_summary()
    
    # Print summary
    print(f"\nEmbedding Summary:")
    print(f"Total facts: {summary.get('total_facts', 0)}")
    print(f"Facts with embeddings: {summary.get('facts_with_embeddings', 0)}")
    print(f"Embedding coverage: {summary.get('embedding_coverage', 0):.2%}")
    print(f"Average confidence: {summary.get('avg_confidence', 0):.2f}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
