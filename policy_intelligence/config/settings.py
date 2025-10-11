"""
Configuration settings for Policy Intelligence Assistant
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Database configurations
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "password",
    "database": "policy_intelligence"
}

QDRANT_CONFIG = {
    "host": "localhost",
    "port": 6333,
    "collection_name": "policy_documents"
}

# NLP Model configurations
SPACY_MODEL = "en_core_web_sm"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NER_MODEL = "gliner"

# Processing parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_TOKENS = 2048

# Entity types for extraction
ENTITY_TYPES = [
    "POLICY", "GO", "JUDGMENT", "COMMISSION_REPORT", 
    "CLAUSE", "SECTION", "DEPARTMENT", "COURT", "DATE"
]

# Relation types
RELATION_TYPES = [
    "refers_to", "influenced_by", "invalidates", "amends",
    "supersedes", "implements", "contradicts", "supports"
]

# Sample documents for testing
SAMPLE_DOCS = [
    "nep_2020_excerpt.pdf",
    "ap_go_education_2022.pdf", 
    "court_judgment_sample.pdf"
]

