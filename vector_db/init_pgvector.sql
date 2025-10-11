-- PostgreSQL Vector Database Initialization for Policy Co-Pilot
-- This script sets up the PostgreSQL database with pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create database if it doesn't exist (run this as superuser)
-- CREATE DATABASE policy;

-- Connect to the policy database
-- \c policy;

-- Main bridge table with vector embeddings
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

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_bridge_doc_id ON bridge_table(doc_id);
CREATE INDEX IF NOT EXISTS idx_bridge_entity_id ON bridge_table(entity_id);
CREATE INDEX IF NOT EXISTS idx_bridge_relation_id ON bridge_table(relation_id);
CREATE INDEX IF NOT EXISTS idx_bridge_span_hash ON bridge_table(span_hash);
CREATE INDEX IF NOT EXISTS idx_bridge_created_at ON bridge_table(created_at);

-- Vector similarity index using ivfflat
CREATE INDEX IF NOT EXISTS idx_bridge_embedding_cosine 
ON bridge_table USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Alternative vector similarity index using hnsw (if available)
-- CREATE INDEX IF NOT EXISTS idx_bridge_embedding_hnsw 
-- ON bridge_table USING hnsw (embedding vector_cosine_ops) 
-- WITH (m = 16, ef_construction = 64);

-- Document metadata table
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

-- Create indexes for document metadata
CREATE INDEX IF NOT EXISTS idx_doc_metadata_doc_id ON document_metadata(doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_metadata_type ON document_metadata(document_type);
CREATE INDEX IF NOT EXISTS idx_doc_metadata_created_at ON document_metadata(created_at);

-- Entity mapping table
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

-- Create indexes for entity mapping
CREATE INDEX IF NOT EXISTS idx_entity_mapping_entity_id ON entity_mapping(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_mapping_type ON entity_mapping(entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_mapping_source ON entity_mapping(source_document);
CREATE INDEX IF NOT EXISTS idx_entity_mapping_confidence ON entity_mapping(confidence);

-- GIN index for JSONB properties
CREATE INDEX IF NOT EXISTS idx_entity_mapping_properties 
ON entity_mapping USING gin (properties);

-- Relation mapping table
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

-- Create indexes for relation mapping
CREATE INDEX IF NOT EXISTS idx_relation_mapping_relation_id ON relation_mapping(relation_id);
CREATE INDEX IF NOT EXISTS idx_relation_mapping_head ON relation_mapping(head_entity_id);
CREATE INDEX IF NOT EXISTS idx_relation_mapping_tail ON relation_mapping(tail_entity_id);
CREATE INDEX IF NOT EXISTS idx_relation_mapping_type ON relation_mapping(relation_type);
CREATE INDEX IF NOT EXISTS idx_relation_mapping_source ON relation_mapping(source_document);

-- GIN index for JSONB properties
CREATE INDEX IF NOT EXISTS idx_relation_mapping_properties 
ON relation_mapping USING gin (properties);

-- Document chunks table for storing text chunks
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_id TEXT UNIQUE NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_embedding VECTOR(384),
    chunk_start INTEGER,
    chunk_end INTEGER,
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for document chunks
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON document_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_start ON document_chunks(chunk_start);

-- Vector similarity index for chunks
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine 
ON document_chunks USING ivfflat (chunk_embedding vector_cosine_ops) 
WITH (lists = 100);

-- Query logs table for tracking user queries
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(384),
    results_count INTEGER,
    processing_time_ms INTEGER,
    user_session_id TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for query logs
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_query_logs_session ON query_logs(user_session_id);

-- Vector similarity index for query logs
CREATE INDEX IF NOT EXISTS idx_query_logs_embedding_cosine 
ON query_logs USING ivfflat (query_embedding vector_cosine_ops) 
WITH (lists = 100);

-- System configuration table
CREATE TABLE IF NOT EXISTS system_config (
    id SERIAL PRIMARY KEY,
    config_key TEXT UNIQUE NOT NULL,
    config_value TEXT,
    config_type TEXT DEFAULT 'string',
    description TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert default configuration
INSERT INTO system_config (config_key, config_value, config_type, description) VALUES
('embedding_model', 'all-MiniLM-L6-v2', 'string', 'Default embedding model'),
('embedding_dimension', '384', 'integer', 'Embedding vector dimension'),
('similarity_threshold', '0.7', 'float', 'Default similarity threshold'),
('max_results', '10', 'integer', 'Maximum results per query'),
('chunk_size', '512', 'integer', 'Default text chunk size'),
('chunk_overlap', '50', 'integer', 'Default chunk overlap'),
('last_updated', NOW()::text, 'timestamp', 'Last system update')
ON CONFLICT (config_key) DO NOTHING;

-- Create views for common queries

-- View for document statistics
CREATE OR REPLACE VIEW document_stats AS
SELECT 
    dm.document_type,
    COUNT(*) as document_count,
    AVG(dm.text_length) as avg_text_length,
    AVG(dm.word_count) as avg_word_count,
    SUM(dm.chunk_count) as total_chunks,
    COUNT(DISTINCT bt.entity_id) as unique_entities,
    COUNT(DISTINCT bt.relation_id) as unique_relations
FROM document_metadata dm
LEFT JOIN bridge_table bt ON dm.doc_id = bt.doc_id
GROUP BY dm.document_type;

-- View for entity statistics
CREATE OR REPLACE VIEW entity_stats AS
SELECT 
    em.entity_type,
    COUNT(*) as entity_count,
    AVG(em.confidence) as avg_confidence,
    COUNT(DISTINCT em.source_document) as document_count
FROM entity_mapping em
GROUP BY em.entity_type;

-- View for relation statistics
CREATE OR REPLACE VIEW relation_stats AS
SELECT 
    rm.relation_type,
    COUNT(*) as relation_count,
    AVG(rm.confidence) as avg_confidence,
    COUNT(DISTINCT rm.source_document) as document_count
FROM relation_mapping rm
GROUP BY rm.relation_type;

-- Function to get similar spans
CREATE OR REPLACE FUNCTION get_similar_spans(
    query_embedding VECTOR(384),
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    bridge_id UUID,
    doc_id TEXT,
    span_text TEXT,
    entity_id TEXT,
    similarity_score FLOAT,
    confidence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        bt.bridge_id,
        bt.doc_id,
        bt.span_text,
        bt.entity_id,
        1 - (bt.embedding <=> query_embedding) as similarity_score,
        bt.confidence
    FROM bridge_table bt
    WHERE 1 - (bt.embedding <=> query_embedding) > similarity_threshold
    ORDER BY bt.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to get entity context
CREATE OR REPLACE FUNCTION get_entity_context(
    target_entity_id TEXT,
    max_results INTEGER DEFAULT 20
)
RETURNS TABLE (
    bridge_id UUID,
    doc_id TEXT,
    span_text TEXT,
    entity_id TEXT,
    confidence FLOAT,
    source_url TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        bt.bridge_id,
        bt.doc_id,
        bt.span_text,
        bt.entity_id,
        bt.confidence,
        bt.source_url
    FROM bridge_table bt
    WHERE bt.entity_id = target_entity_id
    ORDER BY bt.confidence DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to get document spans
CREATE OR REPLACE FUNCTION get_document_spans(
    target_doc_id TEXT,
    max_results INTEGER DEFAULT 100
)
RETURNS TABLE (
    bridge_id UUID,
    span_start INTEGER,
    span_end INTEGER,
    span_text TEXT,
    entity_id TEXT,
    relation_id TEXT,
    confidence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        bt.bridge_id,
        bt.span_start,
        bt.span_end,
        bt.span_text,
        bt.entity_id,
        bt.relation_id,
        bt.confidence
    FROM bridge_table bt
    WHERE bt.doc_id = target_doc_id
    ORDER BY bt.span_start
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to update document statistics
CREATE OR REPLACE FUNCTION update_document_stats()
RETURNS VOID AS $$
BEGIN
    UPDATE document_metadata 
    SET chunk_count = (
        SELECT COUNT(*) 
        FROM bridge_table bt 
        WHERE bt.doc_id = document_metadata.doc_id
    );
END;
$$ LANGUAGE plpgsql;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to bridge_table
CREATE TRIGGER update_bridge_table_updated_at
    BEFORE UPDATE ON bridge_table
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO policy_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO policy_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO policy_user;

-- Create sample data for testing
INSERT INTO document_metadata (doc_id, filename, document_type, text_length, word_count, source_url) VALUES
('GO_75_2021', 'GO_75_2021.pdf', 'GO', 5000, 800, 'https://goir.ap.gov.in/GO_75_2021.pdf'),
('CIRCULAR_123_2021', 'CIRCULAR_123_2021.pdf', 'CIRCULAR', 3000, 500, 'https://cse.ap.gov.in/CIRCULAR_123_2021.pdf'),
('SCERT_CURRICULUM_2021', 'SCERT_CURRICULUM_2021.pdf', 'CURRICULUM', 8000, 1200, 'https://apscert.gov.in/CURRICULUM_2021.pdf')
ON CONFLICT (doc_id) DO NOTHING;

-- Insert sample bridge table entries
INSERT INTO bridge_table (doc_id, span_text, entity_id, embedding, confidence, source_url) VALUES
('GO_75_2021', 'Government Order No. 75 of 2021 amends Rule 5 of the Education Act', 'GO_75_2021_1', '[0.1,0.2,0.3]'::vector, 0.95, 'https://goir.ap.gov.in/GO_75_2021.pdf'),
('CIRCULAR_123_2021', 'Circular No. 123 implements the National Education Policy guidelines', 'CIRCULAR_123_2021_1', '[0.2,0.3,0.4]'::vector, 0.90, 'https://cse.ap.gov.in/CIRCULAR_123_2021.pdf'),
('SCERT_CURRICULUM_2021', 'SCERT curriculum framework for primary education', 'SCERT_CURRICULUM_2021_1', '[0.3,0.4,0.5]'::vector, 0.85, 'https://apscert.gov.in/CURRICULUM_2021.pdf')
ON CONFLICT (span_hash) DO NOTHING;

-- Insert sample entity mappings
INSERT INTO entity_mapping (entity_id, entity_text, entity_type, source_document, confidence) VALUES
('GO_75_2021_1', 'Government Order No. 75 of 2021', 'GO', 'GO_75_2021.pdf', 0.95),
('CIRCULAR_123_2021_1', 'Circular No. 123', 'CIRCULAR', 'CIRCULAR_123_2021.pdf', 0.90),
('SCERT_CURRICULUM_2021_1', 'SCERT curriculum framework', 'CURRICULUM', 'SCERT_CURRICULUM_2021.pdf', 0.85)
ON CONFLICT (entity_id) DO NOTHING;

-- Insert sample relation mappings
INSERT INTO relation_mapping (relation_id, head_entity_id, tail_entity_id, relation_type, confidence, source_document) VALUES
('REL_1', 'GO_75_2021_1', 'EDUCATION_ACT', 'AMENDS', 0.95, 'GO_75_2021.pdf'),
('REL_2', 'CIRCULAR_123_2021_1', 'NEP_2020', 'IMPLEMENTS', 0.90, 'CIRCULAR_123_2021.pdf'),
('REL_3', 'SCERT_CURRICULUM_2021_1', 'PRIMARY_EDUCATION', 'APPLIES_TO', 0.85, 'SCERT_CURRICULUM_2021.pdf')
ON CONFLICT (relation_id) DO NOTHING;

-- Update system configuration
UPDATE system_config 
SET config_value = NOW()::text, updated_at = NOW() 
WHERE config_key = 'last_updated';

-- Display setup completion message
SELECT 'PostgreSQL Vector Database Setup Complete!' as message;
