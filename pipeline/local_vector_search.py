#!/usr/bin/env python3
"""
Local Vector Search for AP Policy Co-Pilot
Alternative to Weaviate when Docker is not available
Uses FAISS for vector similarity and SQLite for metadata
"""
import os
import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using simple similarity search")

class LocalVectorSearch:
    """Local vector search system using FAISS and SQLite"""
    
    def __init__(self, db_path: str = "data/vector_search"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
        # Database and index files
        self.sqlite_db = self.db_path / "facts.db"
        self.vector_index_file = self.db_path / "vector_index.faiss"
        self.embeddings_file = self.db_path / "embeddings.pkl"
        
        # Initialize database
        self._init_database()
        
        # Load or initialize vector index
        self.vector_index = None
        self.fact_id_mapping = {}  # Maps index position to fact_id
        self._load_vector_index()
        
        logger.info("Local vector search initialized")
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        
        # Create facts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                indicator TEXT,
                category TEXT,
                district TEXT,
                year TEXT,
                value REAL,
                unit TEXT,
                source TEXT,
                page_ref INTEGER,
                confidence REAL,
                span_text TEXT,
                pdf_name TEXT,
                search_text TEXT,
                created_at TEXT
            )
        ''')
        
        # Create indexes for faster querying
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicator ON facts(indicator)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_district ON facts(district)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_year ON facts(year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON facts(source)')
        
        conn.commit()
        conn.close()
        
        logger.info("SQLite database initialized")
    
    def _load_vector_index(self):
        """Load existing vector index or create empty one"""
        if FAISS_AVAILABLE and self.vector_index_file.exists():
            try:
                self.vector_index = faiss.read_index(str(self.vector_index_file))
                
                # Load fact ID mapping
                if self.embeddings_file.exists():
                    with open(self.embeddings_file, 'rb') as f:
                        data = pickle.load(f)
                        self.fact_id_mapping = data.get('fact_id_mapping', {})
                
                logger.info(f"Loaded vector index with {self.vector_index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load vector index: {e}")
                self._create_empty_index()
        else:
            self._create_empty_index()
    
    def _create_empty_index(self):
        """Create empty vector index"""
        if FAISS_AVAILABLE:
            self.vector_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for similarity
            logger.info("Created empty FAISS index")
        else:
            self.vector_index = None
            logger.info("FAISS not available, will use simple search")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def load_facts(self, facts_file: str) -> bool:
        """Load facts from JSON file"""
        logger.info(f"Loading facts from {facts_file}")
        
        try:
            with open(facts_file, 'r', encoding='utf-8') as f:
                facts = json.load(f)
        except FileNotFoundError:
            logger.error(f"Facts file not found: {facts_file}")
            return False
        
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM facts')
        
        # Prepare for vector indexing
        embeddings = []
        fact_ids = []
        
        logger.info(f"Processing {len(facts)} facts...")
        
        for i, fact in enumerate(facts):
            try:
                # Create search text
                search_text = self._create_search_text(fact)
                
                # Generate embedding
                embedding = self.generate_embedding(search_text)
                embeddings.append(embedding)
                fact_ids.append(fact['fact_id'])
                
                # Insert into SQLite
                cursor.execute('''
                    INSERT INTO facts (
                        fact_id, indicator, category, district, year, value, unit,
                        source, page_ref, confidence, span_text, pdf_name, 
                        search_text, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fact.get('fact_id', ''),
                    fact.get('indicator', ''),
                    fact.get('category', ''),
                    fact.get('district', ''),
                    fact.get('year', ''),
                    fact.get('value', 0),
                    fact.get('unit', ''),
                    fact.get('source', ''),
                    fact.get('page_ref', 0),
                    fact.get('confidence', 0.8),
                    fact.get('span_text', '')[:1000],  # Limit length
                    fact.get('pdf_name', ''),
                    search_text,
                    fact.get('created_at', datetime.now().isoformat())
                ))
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1}/{len(facts)} facts")
            
            except Exception as e:
                logger.error(f"Failed to process fact {fact.get('fact_id', 'unknown')}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        # Build vector index
        if FAISS_AVAILABLE and embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Create new index
            self.vector_index = faiss.IndexFlatIP(self.embedding_dim)
            self.vector_index.add(embeddings_array)
            
            # Create fact ID mapping
            self.fact_id_mapping = {i: fact_id for i, fact_id in enumerate(fact_ids)}
            
            # Save index and mapping
            faiss.write_index(self.vector_index, str(self.vector_index_file))
            
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump({
                    'fact_id_mapping': self.fact_id_mapping,
                    'embeddings_count': len(embeddings)
                }, f)
            
            logger.info(f"Built vector index with {len(embeddings)} embeddings")
        
        logger.info(f"Loaded {len(facts)} facts successfully")
        return True
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        hybrid_alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search facts using hybrid vector + keyword search
        
        Args:
            query: Search query
            limit: Maximum results
            filters: SQL filters (indicator, district, year, etc.)
            hybrid_alpha: Weight for vector search (0-1, 1=pure vector, 0=pure keyword)
        """
        try:
            results = []
            
            if FAISS_AVAILABLE and self.vector_index and self.vector_index.ntotal > 0:
                # Vector search
                vector_results = self._vector_search(query, limit * 2)  # Get more for reranking
                
                # Keyword search
                keyword_results = self._keyword_search(query, limit * 2, filters)
                
                # Combine results with hybrid scoring
                results = self._combine_hybrid_results(
                    vector_results, keyword_results, hybrid_alpha, limit
                )
            else:
                # Fallback to keyword search only
                results = self._keyword_search(query, limit, filters)
            
            logger.info(f"Found {len(results)} results for query: '{query}'")
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Vector similarity search"""
        if not self.vector_index or self.vector_index.ntotal == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query).astype('float32')
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search vector index
            scores, indices = self.vector_index.search(query_embedding, min(limit, self.vector_index.ntotal))
            
            # Get fact details from SQLite
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                fact_id = self.fact_id_mapping.get(idx)
                if not fact_id:
                    continue
                
                cursor.execute('SELECT * FROM facts WHERE fact_id = ?', (fact_id,))
                row = cursor.fetchone()
                
                if row:
                    result = self._row_to_dict(row)
                    result['vector_score'] = float(score)
                    results.append(result)
            
            conn.close()
            return results
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _keyword_search(self, query: str, limit: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Keyword search using SQLite FTS or LIKE"""
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            # Build query
            where_conditions = []
            params = []
            
            # Text search
            where_conditions.append('(search_text LIKE ? OR span_text LIKE ?)')
            query_pattern = f'%{query}%'
            params.extend([query_pattern, query_pattern])
            
            # Add filters
            if filters:
                if 'indicator' in filters:
                    where_conditions.append('indicator = ?')
                    params.append(filters['indicator'])
                
                if 'district' in filters:
                    where_conditions.append('district = ?')
                    params.append(filters['district'])
                
                if 'year' in filters:
                    where_conditions.append('year = ?')
                    params.append(filters['year'])
                
                if 'category' in filters:
                    where_conditions.append('category = ?')
                    params.append(filters['category'])
            
            # Execute query
            where_clause = ' AND '.join(where_conditions) if where_conditions else '1=1'
            sql = f'''
                SELECT * FROM facts 
                WHERE {where_clause}
                ORDER BY confidence DESC
                LIMIT ?
            '''
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = self._row_to_dict(row)
                result['keyword_score'] = 1.0  # Default score
                results.append(result)
            
            conn.close()
            return results
        
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _combine_hybrid_results(
        self, 
        vector_results: List[Dict[str, Any]], 
        keyword_results: List[Dict[str, Any]], 
        alpha: float, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Combine vector and keyword results with hybrid scoring"""
        
        # Create combined results dictionary
        combined = {}
        
        # Add vector results
        for result in vector_results:
            fact_id = result['fact_id']
            combined[fact_id] = result.copy()
            combined[fact_id]['hybrid_score'] = alpha * result.get('vector_score', 0)
        
        # Add keyword results
        for result in keyword_results:
            fact_id = result['fact_id']
            if fact_id in combined:
                # Combine scores
                combined[fact_id]['hybrid_score'] += (1 - alpha) * result.get('keyword_score', 0)
            else:
                # New result
                combined[fact_id] = result.copy()
                combined[fact_id]['hybrid_score'] = (1 - alpha) * result.get('keyword_score', 0)
        
        # Sort by hybrid score and return top results
        sorted_results = sorted(combined.values(), key=lambda x: x.get('hybrid_score', 0), reverse=True)
        return sorted_results[:limit]
    
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
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary"""
        columns = [
            'fact_id', 'indicator', 'category', 'district', 'year', 'value', 
            'unit', 'source', 'page_ref', 'confidence', 'span_text', 
            'pdf_name', 'search_text', 'created_at'
        ]
        
        result = {}
        for i, col in enumerate(columns):
            if i < len(row):
                result[col] = row[i]
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.sqlite_db)
            cursor = conn.cursor()
            
            # Total facts
            cursor.execute('SELECT COUNT(*) FROM facts')
            total_facts = cursor.fetchone()[0]
            
            # Unique indicators
            cursor.execute('SELECT COUNT(DISTINCT indicator) FROM facts')
            unique_indicators = cursor.fetchone()[0]
            
            # Unique districts
            cursor.execute('SELECT COUNT(DISTINCT district) FROM facts')
            unique_districts = cursor.fetchone()[0]
            
            # Unique years
            cursor.execute('SELECT COUNT(DISTINCT year) FROM facts')
            unique_years = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_facts": total_facts,
                "unique_indicators": unique_indicators,
                "unique_districts": unique_districts,
                "unique_years": unique_years,
                "vector_index_size": self.vector_index.ntotal if self.vector_index else 0,
                "database": "Local SQLite + FAISS" if FAISS_AVAILABLE else "Local SQLite"
            }
        
        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            return {}

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Local vector search system')
    parser.add_argument('--facts-file', default='data/processed/processed_facts.json',
                       help='Processed facts JSON file')
    parser.add_argument('--db-path', default='data/vector_search',
                       help='Database path')
    parser.add_argument('--test-query', help='Test query to run')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize search system
    search_system = LocalVectorSearch(db_path=args.db_path)
    
    # Load facts
    if os.path.exists(args.facts_file):
        success = search_system.load_facts(args.facts_file)
        
        if success:
            stats = search_system.get_statistics()
            print(f"\n✅ Local vector search loaded successfully!")
            print(f"📊 Total facts: {stats.get('total_facts', 0):,}")
            print(f"📈 Unique indicators: {stats.get('unique_indicators', 0)}")
            print(f"🏛️ Unique districts: {stats.get('unique_districts', 0)}")
            print(f"📅 Unique years: {stats.get('unique_years', 0)}")
            print(f"🔍 Vector index size: {stats.get('vector_index_size', 0):,}")
            
            # Test search if query provided
            if args.test_query:
                print(f"\n🔍 Testing search: '{args.test_query}'")
                results = search_system.search(args.test_query, limit=5)
                
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. {result['indicator']} in {result['district']} ({result['year']})")
                    print(f"   Value: {result['value']} {result['unit']}")
                    print(f"   Source: {result['source']}")
                    print(f"   Score: {result.get('hybrid_score', 'N/A'):.3f}")
        else:
            print("❌ Failed to load facts")
    else:
        print(f"❌ Facts file not found: {args.facts_file}")

if __name__ == "__main__":
    main()