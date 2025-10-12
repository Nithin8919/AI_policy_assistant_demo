#!/usr/bin/env python3
"""
Legal-Aware Chunking System for AP Policy Co-Pilot
Advanced document processing with legal structure awareness and Weaviate integration
"""
import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Core libraries
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy

# Weaviate integration
import weaviate
from weaviate.classes.config import Property, DataType

# Legal document processing
from transformers import AutoTokenizer, AutoModel
import torch

logger = logging.getLogger(__name__)

@dataclass
class LegalChunk:
    """Represents a semantically meaningful chunk of legal document"""
    chunk_id: str
    document_id: str
    page_number: int
    section_type: str  # 'header', 'article', 'clause', 'table', 'definition', 'procedure'
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    legal_entities: List[str] = None
    keywords: List[str] = None
    confidence_score: float = 0.0

class LegalDocumentParser:
    """Advanced parser for legal document structure"""
    
    def __init__(self):
        # Load spaCy model for legal text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic processing")
            self.nlp = None
        
        # Legal document patterns
        self.legal_patterns = {
            'section_header': re.compile(r'^(\d+\.?\d*\.?\d*)\s+(.+)$', re.MULTILINE),
            'article_header': re.compile(r'^Article\s+(\d+):?\s*(.+)$', re.MULTILINE | re.IGNORECASE),
            'clause_header': re.compile(r'^\([a-z]\)\s+(.+)$', re.MULTILINE),
            'definition': re.compile(r'^"([^"]+)"\s+means?\s+(.+)$', re.MULTILINE | re.IGNORECASE),
            'procedure': re.compile(r'^(Step\s+\d+|Procedure|Process):\s*(.+)$', re.MULTILINE | re.IGNORECASE),
            'table_header': re.compile(r'^Table\s+\d+[:\s]*(.+)$', re.MULTILINE | re.IGNORECASE),
            'budget_item': re.compile(r'^(\d+\.?\d*)\s+(.+?)\s+(\d+,\d+|\d+)$', re.MULTILINE),
            'enrollment_data': re.compile(r'^(.+?)\s+(\d+)\s+(boys|girls|total)$', re.MULTILINE | re.IGNORECASE)
        }
        
        # Legal entity types
        self.legal_entities = {
            'GOVERNMENT_AGENCIES': ['CSE', 'SCERT', 'UDISE', 'MHRD', 'GOI', 'AP Government'],
            'EDUCATION_TERMS': ['GER', 'NER', 'Dropout Rate', 'Enrollment', 'Retention', 'Transition'],
            'LEGAL_TERMS': ['Act', 'Rule', 'Regulation', 'Policy', 'Guideline', 'Order', 'Circular'],
            'ADMINISTRATIVE_UNITS': ['District', 'Mandal', 'Block', 'Village', 'School', 'Cluster']
        }
    
    def parse_document(self, document_text: str, doc_id: str) -> List[LegalChunk]:
        """Parse document into legal-aware chunks"""
        chunks = []
        
        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(document_text)
        
        chunk_counter = 0
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # Determine section type
            section_type = self._classify_section_type(paragraph)
            
            # Extract legal entities and keywords
            entities = self._extract_legal_entities(paragraph)
            keywords = self._extract_keywords(paragraph)
            
            # Create chunk
            chunk_id = f"{doc_id}_chunk_{chunk_counter:04d}"
            chunk = LegalChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                page_number=self._extract_page_number(paragraph),
                section_type=section_type,
                content=paragraph.strip(),
                metadata={
                    'paragraph_index': para_idx,
                    'word_count': len(paragraph.split()),
                    'char_count': len(paragraph),
                    'has_numbers': bool(re.search(r'\d+', paragraph)),
                    'has_legal_terms': any(term in paragraph.lower() for term in ['act', 'rule', 'policy', 'order']),
                    'extraction_method': 'legal_parser'
                },
                legal_entities=entities,
                keywords=keywords,
                confidence_score=self._calculate_confidence(paragraph, section_type)
            )
            
            chunks.append(chunk)
            chunk_counter += 1
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs"""
        # Split by double newlines first
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split long paragraphs
        refined_paragraphs = []
        for para in paragraphs:
            if len(para.split()) > 200:  # Long paragraph
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len((current_chunk + sentence).split()) > 150:
                        if current_chunk:
                            refined_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence
                
                if current_chunk:
                    refined_paragraphs.append(current_chunk.strip())
            else:
                refined_paragraphs.append(para.strip())
        
        return [p for p in refined_paragraphs if p.strip()]
    
    def _classify_section_type(self, text: str) -> str:
        """Classify the type of legal section"""
        text_lower = text.lower()
        
        # Check patterns
        for pattern_name, pattern in self.legal_patterns.items():
            if pattern.search(text):
                if 'header' in pattern_name:
                    return 'header'
                elif 'article' in pattern_name:
                    return 'article'
                elif 'clause' in pattern_name:
                    return 'clause'
                elif 'definition' in pattern_name:
                    return 'definition'
                elif 'procedure' in pattern_name:
                    return 'procedure'
                elif 'table' in pattern_name:
                    return 'table'
                elif 'budget' in pattern_name:
                    return 'budget_data'
                elif 'enrollment' in pattern_name:
                    return 'enrollment_data'
        
        # Fallback classification
        if any(word in text_lower for word in ['definition', 'means', 'refers to']):
            return 'definition'
        elif any(word in text_lower for word in ['procedure', 'process', 'steps']):
            return 'procedure'
        elif any(word in text_lower for word in ['budget', 'allocation', 'funds', 'rupees']):
            return 'budget_data'
        elif any(word in text_lower for word in ['enrollment', 'students', 'schools', 'teachers']):
            return 'enrollment_data'
        elif len(text.split()) < 20:
            return 'header'
        else:
            return 'content'
    
    def _extract_legal_entities(self, text: str) -> List[str]:
        """Extract legal entities from text"""
        entities = []
        
        # Check for known legal entities
        for entity_type, entity_list in self.legal_entities.items():
            for entity in entity_list:
                if entity.lower() in text.lower():
                    entities.append(entity)
        
        # Use spaCy NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'GPE', 'LAW']:
                    entities.append(ent.text)
        
        return list(set(entities))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Return top keywords by frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:10]
    
    def _extract_page_number(self, text: str) -> int:
        """Extract page number from text"""
        page_match = re.search(r'page\s+(\d+)', text.lower())
        if page_match:
            return int(page_match.group(1))
        return 1  # Default page number
    
    def _calculate_confidence(self, text: str, section_type: str) -> float:
        """Calculate confidence score for chunk quality"""
        score = 0.5  # Base score
        
        # Length factor
        word_count = len(text.split())
        if 20 <= word_count <= 200:
            score += 0.2
        elif word_count > 200:
            score += 0.1
        
        # Content quality factors
        if re.search(r'\d+', text):  # Has numbers
            score += 0.1
        
        if any(term in text.lower() for term in ['policy', 'education', 'school', 'student']):
            score += 0.1
        
        if section_type in ['definition', 'procedure', 'budget_data', 'enrollment_data']:
            score += 0.1
        
        return min(score, 1.0)

class LegalAwareChunker:
    """Main class for legal-aware chunking with Weaviate integration"""
    
    def __init__(self, weaviate_url: str = None, weaviate_api_key: str = None):
        self.parser = LegalDocumentParser()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Weaviate client
        self.weaviate_client = None
        if weaviate_url:
            self._init_weaviate(weaviate_url, weaviate_api_key)
    
    def _init_weaviate(self, url: str, api_key: str = None):
        """Initialize Weaviate client"""
        try:
            if api_key:
                self.weaviate_client = weaviate.Client(
                    url=url,
                    auth_client_secret=weaviate.AuthApiKey(api_key)
                )
            else:
                self.weaviate_client = weaviate.Client(url=url)
            
            logger.info(f"Connected to Weaviate at {url}")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            self.weaviate_client = None
    
    def process_documents(self, input_dir: str = "data/extracted") -> List[LegalChunk]:
        """Process all extracted documents with legal-aware chunking"""
        input_path = Path(input_dir)
        all_chunks = []
        
        # Load extracted data
        extracted_file = input_path / "all_extracted_data.json"
        if not extracted_file.exists():
            logger.error(f"Extracted data file not found: {extracted_file}")
            return []
        
        with open(extracted_file, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        logger.info(f"Processing {len(extracted_data)} documents...")
        
        for doc_name, facts in extracted_data.items():
            logger.info(f"Processing document: {doc_name}")
            
            # Combine all text from the document
            document_text = self._combine_document_text(facts)
            
            # Parse into legal chunks
            chunks = self.parser.parse_document(document_text, doc_name)
            
            # Generate embeddings
            self._generate_embeddings(chunks)
            
            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {doc_name}")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _combine_document_text(self, facts: List[Dict[str, Any]]) -> str:
        """Combine all text from document facts"""
        text_parts = []
        
        for fact in facts:
            text = fact.get('text', '').strip()
            if text and len(text) > 10:  # Only meaningful text
                text_parts.append(text)
        
        return '\n\n'.join(text_parts)
    
    def _generate_embeddings(self, chunks: List[LegalChunk]):
        """Generate embeddings for chunks"""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
    
    def save_to_weaviate(self, chunks: List[LegalChunk], collection_name: str = "LegalDocuments"):
        """Save chunks to Weaviate"""
        if not self.weaviate_client:
            logger.error("Weaviate client not initialized")
            return False
        
        try:
            # Create collection if it doesn't exist
            if not self.weaviate_client.schema.exists(collection_name):
                self._create_weaviate_schema(collection_name)
            
            # Batch insert chunks
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self._insert_batch(batch, collection_name)
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            logger.info(f"Successfully saved {len(chunks)} chunks to Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to Weaviate: {e}")
            return False
    
    def _create_weaviate_schema(self, collection_name: str):
        """Create Weaviate schema for legal documents"""
        schema = {
            "class": collection_name,
            "description": "Legal document chunks with semantic search capabilities",
            "properties": [
                {
                    "name": "chunk_id",
                    "dataType": ["string"],
                    "description": "Unique identifier for the chunk"
                },
                {
                    "name": "document_id",
                    "dataType": ["string"],
                    "description": "Source document identifier"
                },
                {
                    "name": "page_number",
                    "dataType": ["int"],
                    "description": "Page number in source document"
                },
                {
                    "name": "section_type",
                    "dataType": ["string"],
                    "description": "Type of legal section (header, article, clause, etc.)"
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The actual text content of the chunk"
                },
                {
                    "name": "legal_entities",
                    "dataType": ["string[]"],
                    "description": "Legal entities found in the chunk"
                },
                {
                    "name": "keywords",
                    "dataType": ["string[]"],
                    "description": "Important keywords extracted from the chunk"
                },
                {
                    "name": "confidence_score",
                    "dataType": ["number"],
                    "description": "Confidence score for chunk quality"
                },
                {
                    "name": "word_count",
                    "dataType": ["int"],
                    "description": "Number of words in the chunk"
                },
                {
                    "name": "has_legal_terms",
                    "dataType": ["boolean"],
                    "description": "Whether chunk contains legal terminology"
                },
                {
                    "name": "created_at",
                    "dataType": ["date"],
                    "description": "Timestamp when chunk was created"
                }
            ],
            "vectorizer": "none"
        }
        
        self.weaviate_client.schema.create_class(schema)
        logger.info(f"Created Weaviate schema: {collection_name}")
    
    def _insert_batch(self, chunks: List[LegalChunk], collection_name: str):
        """Insert a batch of chunks into Weaviate"""
        with self.weaviate_client.batch as batch:
            for chunk in chunks:
                properties = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "page_number": chunk.page_number,
                    "section_type": chunk.section_type,
                    "content": chunk.content,
                    "legal_entities": chunk.legal_entities or [],
                    "keywords": chunk.keywords or [],
                    "confidence_score": chunk.confidence_score,
                    "word_count": chunk.metadata.get('word_count', 0),
                    "has_legal_terms": chunk.metadata.get('has_legal_terms', False),
                    "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                }
                
                batch.add_data_object(
                    data_object=properties,
                    class_name=collection_name,
                    vector=chunk.embedding.tolist() if chunk.embedding is not None else None
                )
    
    def search_legal_documents(self, query: str, limit: int = 10, 
                             section_types: List[str] = None,
                             min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Search legal documents with advanced filtering"""
        if not self.weaviate_client:
            logger.error("Weaviate client not initialized")
            return []
        
        try:
            # Build query with text search
            query_builder = self.weaviate_client.query.get(
                "LegalDocuments",
                ["chunk_id", "document_id", "content", "section_type", 
                 "legal_entities", "keywords", "confidence_score"]
            )
            
            # Add text search filter
            query_builder = query_builder.with_where({
                "path": ["content"],
                "operator": "Like",
                "valueText": f"*{query}*"
            })
            
            # Add filters
            if section_types:
                query_builder = query_builder.with_where({
                    "path": ["section_type"],
                    "operator": "ContainsAny",
                    "valueText": section_types
                })
            
            if min_confidence > 0:
                query_builder = query_builder.with_where({
                    "path": ["confidence_score"],
                    "operator": "GreaterThan",
                    "valueNumber": min_confidence
                })
            
            # Execute query
            result = query_builder.with_limit(limit).do()
            
            # Process results
            chunks = []
            if "data" in result and "Get" in result["data"]:
                for item in result["data"]["Get"]["LegalDocuments"]:
                    chunks.append({
                        "chunk_id": item["chunk_id"],
                        "document_id": item["document_id"],
                        "content": item["content"],
                        "section_type": item["section_type"],
                        "legal_entities": item["legal_entities"],
                        "keywords": item["keywords"],
                        "confidence_score": item["confidence_score"]
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

def main():
    """Main function to run legal-aware chunking"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Legal-aware chunking with Weaviate integration')
    parser.add_argument('--weaviate-url', help='Weaviate server URL')
    parser.add_argument('--weaviate-api-key', help='Weaviate API key')
    parser.add_argument('--collection-name', default='LegalDocuments', help='Weaviate collection name')
    parser.add_argument('--input-dir', default='data/extracted', help='Input directory with extracted data')
    parser.add_argument('--test-search', action='store_true', help='Test search functionality')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize chunker
    chunker = LegalAwareChunker(
        weaviate_url=args.weaviate_url,
        weaviate_api_key=args.weaviate_api_key
    )
    
    if args.test_search:
        # Test search functionality
        print("🔍 Testing legal document search...")
        results = chunker.search_legal_documents(
            "school enrollment policy",
            limit=5,
            section_types=["policy", "procedure"],
            min_confidence=0.3
        )
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['section_type']}] {result['content'][:100]}...")
            print(f"   Confidence: {result['confidence_score']:.3f}")
            print(f"   Entities: {result['legal_entities']}")
            print()
    else:
        # Process documents
        print("🚀 Starting legal-aware chunking...")
        chunks = chunker.process_documents(args.input_dir)
        
        if chunks:
            print(f"✅ Created {len(chunks)} legal-aware chunks")
            
            # Save to Weaviate if configured
            if chunker.weaviate_client:
                print("💾 Saving to Weaviate...")
                success = chunker.save_to_weaviate(chunks, args.collection_name)
                if success:
                    print("✅ Successfully saved to Weaviate!")
                else:
                    print("❌ Failed to save to Weaviate")
            else:
                print("⚠️ Weaviate not configured, skipping save")
        else:
            print("❌ No chunks created")

if __name__ == "__main__":
    main()
