"""
PostgreSQL Vector Database Embedding Loader
"""
import psycopg2
import psycopg2.extras
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingChunk:
    """Embedding chunk structure"""
    doc_id: str
    chunk_text: str
    chunk_start: int
    chunk_end: int
    embedding: List[float]
    chunk_id: str
    word_count: int

class PGVectorLoader:
    """PostgreSQL vector database loader"""
    
    def __init__(self, db_config: Optional[Dict[str, str]] = None):
        self.db_config = db_config or {
            'host': 'localhost',
            'port': '5432',
            'database': 'policy',
            'user': 'postgres',
            'password': '1234'
        }
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Test database connection
        self._test_connection()
    
    def _test_connection(self):
        """Test PostgreSQL connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            logger.info("PostgreSQL connection successful")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
    
    def load_document_embeddings(self, document_data: Dict[str, Any]) -> int:
        """
        Load document embeddings into PostgreSQL
        
        Args:
            document_data: Document data with text and metadata
            
        Returns:
            Number of chunks loaded
        """
        try:
            doc_id = document_data['metadata']['doc_id']
            text = document_data['text']
            
            logger.info(f"Loading embeddings for document: {doc_id}")
            
            # Chunk the text
            chunks = self._chunk_text(text, doc_id)
            
            # Generate embeddings for chunks
            embeddings = self._generate_embeddings([chunk.chunk_text for chunk in chunks])
            
            # Update chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()
            
            # Load chunks into database
            loaded_count = self._load_chunks_to_db(chunks, document_data['metadata'])
            
            logger.info(f"Loaded {loaded_count} chunks for document {doc_id}")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load document embeddings: {e}")
            return 0
    
    def _chunk_text(self, text: str, doc_id: str, chunk_size: int = 512, overlap: int = 50) -> List[EmbeddingChunk]:
        """
        Chunk text into smaller pieces for embedding
        
        Args:
            text: Input text
            doc_id: Document identifier
            chunk_size: Size of each chunk in words
            overlap: Overlap between chunks in words
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk = EmbeddingChunk(
                doc_id=doc_id,
                chunk_text=chunk_text,
                chunk_start=i,
                chunk_end=i + len(chunk_words),
                embedding=[],  # Will be filled later
                chunk_id=f"{doc_id}_chunk_{i//(chunk_size - overlap)}",
                word_count=len(chunk_words)
            )
            
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def _load_chunks_to_db(self, chunks: List[EmbeddingChunk], metadata: Dict[str, Any]) -> int:
        """
        Load chunks into PostgreSQL database
        
        Args:
            chunks: List of embedding chunks
            metadata: Document metadata
            
        Returns:
            Number of chunks loaded
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            loaded_count = 0
            
            for chunk in chunks:
                try:
                    # Convert embedding to PostgreSQL vector format
                    embedding_str = '[' + ','.join(map(str, chunk.embedding)) + ']'
                    
                    # Insert into document_chunks table
                    cursor.execute("""
                        INSERT INTO document_chunks 
                        (doc_id, chunk_id, chunk_text, chunk_embedding, chunk_start, chunk_end, word_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            chunk_text = EXCLUDED.chunk_text,
                            chunk_embedding = EXCLUDED.chunk_embedding,
                            chunk_start = EXCLUDED.chunk_start,
                            chunk_end = EXCLUDED.chunk_end,
                            word_count = EXCLUDED.word_count
                    """, (
                        chunk.doc_id,
                        chunk.chunk_id,
                        chunk.chunk_text,
                        embedding_str,
                        chunk.chunk_start,
                        chunk.chunk_end,
                        chunk.word_count
                    ))
                    
                    # Insert into bridge_table
                    span_hash = hashlib.sha256(chunk.chunk_text.encode()).hexdigest()
                    
                    cursor.execute("""
                        INSERT INTO bridge_table 
                        (doc_id, span_start, span_end, span_text, span_hash, embedding, confidence, source_url)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (span_hash) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            confidence = EXCLUDED.confidence
                    """, (
                        chunk.doc_id,
                        chunk.chunk_start,
                        chunk.chunk_end,
                        chunk.chunk_text,
                        span_hash,
                        embedding_str,
                        1.0,
                        metadata.get('source_url')
                    ))
                    
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to load chunk {chunk.chunk_id}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Loaded {loaded_count} chunks to database")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load chunks to database: {e}")
            return 0
    
    def load_entity_embeddings(self, entities: List[Dict[str, Any]], doc_id: str) -> int:
        """
        Load entity embeddings into bridge table
        
        Args:
            entities: List of entity data
            doc_id: Document identifier
            
        Returns:
            Number of entities loaded
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            loaded_count = 0
            
            for entity in entities:
                try:
                    entity_text = entity['text']
                    entity_id = entity['entity_id']
                    
                    # Generate embedding for entity text
                    embedding = self.embedding_model.encode([entity_text])
                    embedding_str = '[' + ','.join(map(str, embedding[0])) + ']'
                    
                    # Update bridge table with entity information
                    cursor.execute("""
                        UPDATE bridge_table 
                        SET entity_id = %s, confidence = %s
                        WHERE doc_id = %s AND span_text ILIKE %s
                    """, (
                        entity_id,
                        entity['confidence'],
                        doc_id,
                        f"%{entity_text}%"
                    ))
                    
                    # Insert entity mapping
                    cursor.execute("""
                        INSERT INTO entity_mapping 
                        (entity_id, entity_text, entity_type, source_document, start_position, end_position, confidence)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (entity_id) DO UPDATE SET
                            entity_text = EXCLUDED.entity_text,
                            entity_type = EXCLUDED.entity_type,
                            source_document = EXCLUDED.source_document,
                            start_position = EXCLUDED.start_position,
                            end_position = EXCLUDED.end_position,
                            confidence = EXCLUDED.confidence
                    """, (
                        entity_id,
                        entity_text,
                        entity['label'],
                        doc_id,
                        entity['start'],
                        entity['end'],
                        entity['confidence']
                    ))
                    
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to load entity {entity.get('entity_id', 'unknown')}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Loaded {loaded_count} entity embeddings")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load entity embeddings: {e}")
            return 0
    
    def load_relation_embeddings(self, relations: List[Dict[str, Any]], doc_id: str) -> int:
        """
        Load relation embeddings into bridge table
        
        Args:
            relations: List of relation data
            doc_id: Document identifier
            
        Returns:
            Number of relations loaded
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            loaded_count = 0
            
            for relation in relations:
                try:
                    relation_id = relation['relation_id']
                    head_entity_id = relation['head_entity_id']
                    tail_entity_id = relation['tail_entity_id']
                    relation_type = relation['relation_type']
                    
                    # Insert relation mapping
                    cursor.execute("""
                        INSERT INTO relation_mapping 
                        (relation_id, head_entity_id, tail_entity_id, relation_type, context, confidence, source_document)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (relation_id) DO UPDATE SET
                            head_entity_id = EXCLUDED.head_entity_id,
                            tail_entity_id = EXCLUDED.tail_entity_id,
                            relation_type = EXCLUDED.relation_type,
                            context = EXCLUDED.context,
                            confidence = EXCLUDED.confidence,
                            source_document = EXCLUDED.source_document
                    """, (
                        relation_id,
                        head_entity_id,
                        tail_entity_id,
                        relation_type,
                        relation.get('context', ''),
                        relation['confidence'],
                        doc_id
                    ))
                    
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to load relation {relation.get('relation_id', 'unknown')}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Loaded {loaded_count} relation embeddings")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load relation embeddings: {e}")
            return 0
    
    def batch_load_from_json(self, json_file: str) -> Dict[str, int]:
        """
        Batch load embeddings from JSON file
        
        Args:
            json_file: Path to JSON file containing document data
            
        Returns:
            Dictionary with loading statistics
        """
        stats = {
            'documents_loaded': 0,
            'chunks_loaded': 0,
            'entities_loaded': 0,
            'relations_loaded': 0,
            'errors': 0
        }
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle single document or list of documents
            documents = data if isinstance(data, list) else [data]
            
            for doc_data in documents:
                try:
                    # Load document embeddings
                    chunks_loaded = self.load_document_embeddings(doc_data)
                    stats['chunks_loaded'] += chunks_loaded
                    
                    # Load entity embeddings if available
                    if 'entities' in doc_data:
                        entities_loaded = self.load_entity_embeddings(
                            doc_data['entities'], 
                            doc_data['metadata']['doc_id']
                        )
                        stats['entities_loaded'] += entities_loaded
                    
                    # Load relation embeddings if available
                    if 'relations' in doc_data:
                        relations_loaded = self.load_relation_embeddings(
                            doc_data['relations'], 
                            doc_data['metadata']['doc_id']
                        )
                        stats['relations_loaded'] += relations_loaded
                    
                    stats['documents_loaded'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to load document {doc_data.get('metadata', {}).get('doc_id', 'unknown')}: {e}")
                    stats['errors'] += 1
            
            logger.info(f"Batch loading completed: {stats}")
            
        except Exception as e:
            logger.error(f"Failed to batch load from JSON: {e}")
            stats['errors'] += 1
        
        return stats
    
    def get_similar_chunks(self, query_text: str, max_results: int = 10, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get similar chunks using vector similarity
        
        Args:
            query_text: Query text
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar chunks
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query_text])
            embedding_str = '[' + ','.join(map(str, query_embedding[0])) + ']'
            
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Query for similar chunks
            cursor.execute("""
                SELECT 
                    bt.bridge_id,
                    bt.doc_id,
                    bt.span_text,
                    bt.entity_id,
                    bt.confidence,
                    bt.source_url,
                    1 - (bt.embedding <=> %s::vector) as similarity_score
                FROM bridge_table bt
                WHERE 1 - (bt.embedding <=> %s::vector) > %s
                ORDER BY bt.embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, similarity_threshold, embedding_str, max_results))
            
            results = cursor.fetchall()
            conn.close()
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get similar chunks: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            stats = {}
            
            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            stats['total_chunks'] = cursor.fetchone()[0]
            
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM document_metadata")
            stats['total_documents'] = cursor.fetchone()[0]
            
            # Count entities
            cursor.execute("SELECT COUNT(*) FROM entity_mapping")
            stats['total_entities'] = cursor.fetchone()[0]
            
            # Count relations
            cursor.execute("SELECT COUNT(*) FROM relation_mapping")
            stats['total_relations'] = cursor.fetchone()[0]
            
            # Average embedding dimension
            cursor.execute("SELECT array_length(embedding, 1) FROM bridge_table LIMIT 1")
            result = cursor.fetchone()
            if result:
                stats['embedding_dimension'] = result[0]
            
            conn.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

def main():
    """Test the PGVector loader"""
    loader = PGVectorLoader()
    
    # Test database connection
    print("Testing PostgreSQL connection...")
    stats = loader.get_database_stats()
    print(f"Database statistics: {stats}")
    
    # Test embedding generation
    print("Testing embedding generation...")
    test_texts = [
        "Government Order No. 75 of 2021 amends Rule 5 of the Education Act",
        "The Right to Education Act applies to all schools in Andhra Pradesh",
        "SCERT implements the National Education Policy guidelines"
    ]
    
    embeddings = loader._generate_embeddings(test_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test similarity search
    print("Testing similarity search...")
    query = "education policy implementation"
    similar_chunks = loader.get_similar_chunks(query, max_results=5)
    print(f"Found {len(similar_chunks)} similar chunks")
    
    for chunk in similar_chunks[:3]:
        print(f"- {chunk['span_text'][:100]}... (similarity: {chunk['similarity_score']:.3f})")

if __name__ == "__main__":
    main()
