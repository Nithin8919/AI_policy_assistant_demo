"""
PostgreSQL Bridge Table Manager with pgvector support
"""
import psycopg2
import psycopg2.extras
from typing import List, Dict, Any, Optional, Union
import logging
import uuid
import hashlib
import json
import os
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class BridgeTableManager:
    """Manages PostgreSQL bridge table with vector embeddings"""
    
    def __init__(self, db_config: Optional[Dict[str, str]] = None):
        self.db_config = db_config or {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'policy'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', '1234')
        }
        
        self._test_connection()
        self._create_tables()
    
    def _test_connection(self):
        """Test PostgreSQL connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            conn.close()
            logger.info("PostgreSQL connection successful")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise
    
    def _create_tables(self):
        """Create bridge table schema with pgvector support"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Main bridge table with vector embeddings
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS bridge_table (
                bridge_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                doc_id TEXT NOT NULL,
                span_start INTEGER,
                span_end INTEGER,
                span_text TEXT NOT NULL,
                span_hash TEXT UNIQUE,
                entity_id TEXT,
                relation_id TEXT,
                embedding VECTOR(384),  -- all-MiniLM-L6-v2 dimension
                confidence FLOAT DEFAULT 1.0,
                version_id TEXT,
                source_url TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """)
            
            # Create indexes for performance
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bridge_doc_id ON bridge_table(doc_id);
            CREATE INDEX IF NOT EXISTS idx_bridge_entity_id ON bridge_table(entity_id);
            CREATE INDEX IF NOT EXISTS idx_bridge_relation_id ON bridge_table(relation_id);
            CREATE INDEX IF NOT EXISTS idx_bridge_embedding ON bridge_table USING ivfflat (embedding vector_cosine_ops);
            CREATE INDEX IF NOT EXISTS idx_bridge_span_hash ON bridge_table(span_hash);
            """)
            
            # Document metadata table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_metadata (
                id SERIAL PRIMARY KEY,
                doc_id TEXT UNIQUE NOT NULL,
                filename TEXT,
                file_path TEXT,
                document_type TEXT,
                text_length INTEGER,
                word_count INTEGER,
                chunk_count INTEGER,
                source_url TEXT,
                processing_date TIMESTAMP DEFAULT NOW(),
                created_at TIMESTAMP DEFAULT NOW()
            );
            """)
            
            # Entity mapping table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_mapping (
                id SERIAL PRIMARY KEY,
                entity_id TEXT UNIQUE NOT NULL,
                entity_text TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                source_document TEXT,
                start_position INTEGER,
                end_position INTEGER,
                confidence FLOAT DEFAULT 1.0,
                properties JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """)
            
            # Relation mapping table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS relation_mapping (
                id SERIAL PRIMARY KEY,
                relation_id TEXT UNIQUE NOT NULL,
                head_entity_id TEXT NOT NULL,
                tail_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                context TEXT,
                confidence FLOAT DEFAULT 1.0,
                source_document TEXT,
                properties JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Bridge table schema created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def insert_span(
        self,
        doc_id: str,
        span_text: str,
        span_start: int,
        span_end: int,
        entity_id: Optional[str] = None,
        relation_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        confidence: float = 1.0,
        source_url: Optional[str] = None,
        version_id: Optional[str] = None
    ) -> str:
        """
        Insert a span into the bridge table
        
        Args:
            doc_id: Document identifier
            span_text: Text content of the span
            span_start: Start position in document
            span_end: End position in document
            entity_id: Associated entity ID
            relation_id: Associated relation ID
            embedding: Vector embedding of the span
            confidence: Confidence score
            source_url: Source URL
            version_id: Version identifier
            
        Returns:
            Bridge ID of the inserted span
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Generate span hash
            span_hash = hashlib.sha256(span_text.encode()).hexdigest()
            
            # Convert embedding to PostgreSQL vector format
            embedding_str = None
            if embedding:
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            query = """
            INSERT INTO bridge_table 
            (doc_id, span_start, span_end, span_text, span_hash, entity_id, relation_id, 
             embedding, confidence, source_url, version_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING bridge_id;
            """
            
            cursor.execute(query, (
                doc_id, span_start, span_end, span_text, span_hash,
                entity_id, relation_id, embedding_str, confidence,
                source_url, version_id
            ))
            
            bridge_id = cursor.fetchone()[0]
            conn.commit()
            conn.close()
            
            logger.debug(f"Inserted span: {bridge_id}")
            return str(bridge_id)
            
        except Exception as e:
            logger.error(f"Failed to insert span: {e}")
            raise
    
    def bulk_insert_spans(
        self,
        spans: List[Dict[str, Any]]
    ) -> int:
        """
        Bulk insert spans into the bridge table
        
        Args:
            spans: List of span dictionaries
            
        Returns:
            Number of spans inserted
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            inserted_count = 0
            
            for span in spans:
                try:
                    # Generate span hash
                    span_text = span.get('span_text', '')
                    span_hash = hashlib.sha256(span_text.encode()).hexdigest()
                    
                    # Convert embedding to PostgreSQL vector format
                    embedding = span.get('embedding')
                    embedding_str = None
                    if embedding:
                        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    query = """
                    INSERT INTO bridge_table 
                    (doc_id, span_start, span_end, span_text, span_hash, entity_id, relation_id, 
                     embedding, confidence, source_url, version_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (span_hash) DO UPDATE SET
                        entity_id = EXCLUDED.entity_id,
                        relation_id = EXCLUDED.relation_id,
                        confidence = EXCLUDED.confidence,
                        updated_at = NOW();
                    """
                    
                    cursor.execute(query, (
                        span.get('doc_id'),
                        span.get('span_start'),
                        span.get('span_end'),
                        span_text,
                        span_hash,
                        span.get('entity_id'),
                        span.get('relation_id'),
                        embedding_str,
                        span.get('confidence', 1.0),
                        span.get('source_url'),
                        span.get('version_id')
                    ))
                    
                    inserted_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to insert span: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Bulk inserted {inserted_count} spans")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Bulk span insertion failed: {e}")
            raise
    
    def add_document_metadata(
        self,
        doc_id: str,
        filename: str,
        file_path: Optional[str] = None,
        document_type: Optional[str] = None,
        text_length: Optional[int] = None,
        word_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        source_url: Optional[str] = None
    ) -> bool:
        """
        Add document metadata
        
        Args:
            doc_id: Document identifier
            filename: Original filename
            file_path: File path
            document_type: Type of document
            text_length: Length of text
            word_count: Word count
            chunk_count: Number of chunks
            source_url: Source URL
            
        Returns:
            True if successful
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            query = """
            INSERT INTO document_metadata 
            (doc_id, filename, file_path, document_type, text_length, word_count, chunk_count, source_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                filename = EXCLUDED.filename,
                file_path = EXCLUDED.file_path,
                document_type = EXCLUDED.document_type,
                text_length = EXCLUDED.text_length,
                word_count = EXCLUDED.word_count,
                chunk_count = EXCLUDED.chunk_count,
                source_url = EXCLUDED.source_url;
            """
            
            cursor.execute(query, (
                doc_id, filename, file_path, document_type,
                text_length, word_count, chunk_count, source_url
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added metadata for document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document metadata: {e}")
            return False
    
    def add_entity_mapping(
        self,
        entity_id: str,
        entity_text: str,
        entity_type: str,
        source_document: Optional[str] = None,
        start_position: Optional[int] = None,
        end_position: Optional[int] = None,
        confidence: float = 1.0,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add entity mapping
        
        Args:
            entity_id: Entity identifier
            entity_text: Entity text
            entity_type: Type of entity
            source_document: Source document
            start_position: Start position
            end_position: End position
            confidence: Confidence score
            properties: Additional properties
            
        Returns:
            True if successful
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            properties_json = json.dumps(properties) if properties else None
            
            query = """
            INSERT INTO entity_mapping 
            (entity_id, entity_text, entity_type, source_document, start_position, end_position, confidence, properties)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_id) DO UPDATE SET
                entity_text = EXCLUDED.entity_text,
                entity_type = EXCLUDED.entity_type,
                source_document = EXCLUDED.source_document,
                start_position = EXCLUDED.start_position,
                end_position = EXCLUDED.end_position,
                confidence = EXCLUDED.confidence,
                properties = EXCLUDED.properties;
            """
            
            cursor.execute(query, (
                entity_id, entity_text, entity_type, source_document,
                start_position, end_position, confidence, properties_json
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Added entity mapping: {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add entity mapping: {e}")
            return False
    
    def add_relation_mapping(
        self,
        relation_id: str,
        head_entity_id: str,
        tail_entity_id: str,
        relation_type: str,
        context: Optional[str] = None,
        confidence: float = 1.0,
        source_document: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add relation mapping
        
        Args:
            relation_id: Relation identifier
            head_entity_id: Head entity ID
            tail_entity_id: Tail entity ID
            relation_type: Type of relation
            context: Context text
            confidence: Confidence score
            source_document: Source document
            properties: Additional properties
            
        Returns:
            True if successful
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            properties_json = json.dumps(properties) if properties else None
            
            query = """
            INSERT INTO relation_mapping 
            (relation_id, head_entity_id, tail_entity_id, relation_type, context, confidence, source_document, properties)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (relation_id) DO UPDATE SET
                head_entity_id = EXCLUDED.head_entity_id,
                tail_entity_id = EXCLUDED.tail_entity_id,
                relation_type = EXCLUDED.relation_type,
                context = EXCLUDED.context,
                confidence = EXCLUDED.confidence,
                source_document = EXCLUDED.source_document,
                properties = EXCLUDED.properties;
            """
            
            cursor.execute(query, (
                relation_id, head_entity_id, tail_entity_id, relation_type,
                context, confidence, source_document, properties_json
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Added relation mapping: {relation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add relation mapping: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge table statistics"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            stats = {}
            
            # Count spans
            cursor.execute("SELECT COUNT(*) FROM bridge_table")
            stats['total_spans'] = cursor.fetchone()[0]
            
            # Count documents
            cursor.execute("SELECT COUNT(*) FROM document_metadata")
            stats['total_documents'] = cursor.fetchone()[0]
            
            # Count entities
            cursor.execute("SELECT COUNT(*) FROM entity_mapping")
            stats['total_entities'] = cursor.fetchone()[0]
            
            # Count relations
            cursor.execute("SELECT COUNT(*) FROM relation_mapping")
            stats['total_relations'] = cursor.fetchone()[0]
            
            # Entity type distribution
            cursor.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM entity_mapping
            GROUP BY entity_type
            ORDER BY count DESC
            """)
            stats['entity_type_distribution'] = dict(cursor.fetchall())
            
            # Document type distribution
            cursor.execute("""
            SELECT document_type, COUNT(*) as count
            FROM document_metadata
            GROUP BY document_type
            ORDER BY count DESC
            """)
            stats['document_type_distribution'] = dict(cursor.fetchall())
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM bridge_table")
            avg_conf = cursor.fetchone()[0]
            stats['average_confidence'] = round(avg_conf, 3) if avg_conf else 0
            
            conn.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the system"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
            SELECT dm.*, COUNT(bt.bridge_id) as span_count
            FROM document_metadata dm
            LEFT JOIN bridge_table bt ON dm.doc_id = bt.doc_id
            GROUP BY dm.id
            ORDER BY dm.created_at DESC
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document details"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
            SELECT dm.*, COUNT(bt.bridge_id) as span_count
            FROM document_metadata dm
            LEFT JOIN bridge_table bt ON dm.doc_id = bt.doc_id
            WHERE dm.doc_id = %s
            GROUP BY dm.id
            """
            
            cursor.execute(query, (doc_id,))
            result = cursor.fetchone()
            
            conn.close()
            
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            return None
    
    def add_document(
        self,
        document_type: str,
        source_url: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> str:
        """
        Add a new document to the system
        
        Args:
            document_type: Type of document
            source_url: Source URL
            file_path: File path
            
        Returns:
            Document ID
        """
        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Extract filename
            filename = Path(file_path).name if file_path else f"{document_type}_{doc_id[:8]}"
            
            # Add metadata
            self.add_document_metadata(
                doc_id=doc_id,
                filename=filename,
                file_path=file_path,
                document_type=document_type,
                source_url=source_url
            )
            
            logger.info(f"Added document: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    def check_connection(self) -> bool:
        """Check if database connection is healthy"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
        except:
            return False
    
    def close(self):
        """Close database connection (no-op for psycopg2)"""
        pass

