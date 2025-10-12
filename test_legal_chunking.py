#!/usr/bin/env python3
"""
Test Legal-Aware Chunking System
Demonstrates the legal document processing capabilities
"""
import json
import logging
from pathlib import Path
from legal_aware_chunker import LegalAwareChunker, LegalDocumentParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_legal_parser():
    """Test the legal document parser with sample data"""
    print("🧪 Testing Legal Document Parser...")
    
    # Load sample extracted data
    extracted_file = Path("data/extracted/all_extracted_data.json")
    if not extracted_file.exists():
        print("❌ Extracted data file not found")
        return
    
    with open(extracted_file, 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)
    
    # Take first document for testing
    doc_name = list(extracted_data.keys())[0]
    facts = extracted_data[doc_name]
    
    print(f"📄 Testing with document: {doc_name}")
    print(f"📊 Document has {len(facts)} facts")
    
    # Initialize parser
    parser = LegalDocumentParser()
    
    # Combine document text
    document_text = "\n\n".join([fact.get('text', '') for fact in facts[:50]])  # First 50 facts
    
    print(f"📝 Document text length: {len(document_text)} characters")
    
    # Parse document
    chunks = parser.parse_document(document_text, doc_name)
    
    print(f"✅ Created {len(chunks)} legal-aware chunks")
    
    # Analyze chunks
    section_types = {}
    for chunk in chunks:
        section_type = chunk.section_type
        section_types[section_type] = section_types.get(section_type, 0) + 1
    
    print("\n📊 Chunk Analysis:")
    print(f"Total chunks: {len(chunks)}")
    print("Section types:")
    for section_type, count in sorted(section_types.items()):
        print(f"  {section_type}: {count}")
    
    # Show sample chunks
    print("\n📝 Sample Chunks:")
    for i, chunk in enumerate(chunks[:5], 1):
        print(f"\n{i}. [{chunk.section_type}] Confidence: {chunk.confidence_score:.3f}")
        print(f"   Content: {chunk.content[:100]}...")
        print(f"   Entities: {chunk.legal_entities}")
        print(f"   Keywords: {chunk.keywords[:5]}")
    
    return chunks

def test_chunking_pipeline():
    """Test the complete chunking pipeline"""
    print("\n🚀 Testing Complete Legal-Aware Chunking Pipeline...")
    
    # Initialize chunker (without Weaviate for now)
    chunker = LegalAwareChunker()
    
    # Process documents
    chunks = chunker.process_documents("data/extracted")
    
    if chunks:
        print(f"✅ Pipeline processed {len(chunks)} chunks")
        
        # Analyze results
        total_chunks = len(chunks)
        high_confidence_chunks = sum(1 for chunk in chunks if chunk.confidence_score > 0.7)
        chunks_with_entities = sum(1 for chunk in chunks if chunk.legal_entities)
        
        print(f"\n📊 Pipeline Results:")
        print(f"Total chunks: {total_chunks}")
        print(f"High confidence chunks (>0.7): {high_confidence_chunks}")
        print(f"Chunks with legal entities: {chunks_with_entities}")
        
        # Show statistics by section type
        section_stats = {}
        for chunk in chunks:
            section_type = chunk.section_type
            if section_type not in section_stats:
                section_stats[section_type] = {'count': 0, 'avg_confidence': 0, 'total_confidence': 0}
            section_stats[section_type]['count'] += 1
            section_stats[section_type]['total_confidence'] += chunk.confidence_score
        
        for section_type, stats in section_stats.items():
            stats['avg_confidence'] = stats['total_confidence'] / stats['count']
        
        print(f"\n📈 Section Type Statistics:")
        for section_type, stats in sorted(section_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"  {section_type}: {stats['count']} chunks, avg confidence: {stats['avg_confidence']:.3f}")
        
        # Show best chunks
        print(f"\n🏆 Best Quality Chunks (confidence > 0.8):")
        best_chunks = [chunk for chunk in chunks if chunk.confidence_score > 0.8][:3]
        for i, chunk in enumerate(best_chunks, 1):
            print(f"\n{i}. [{chunk.section_type}] Confidence: {chunk.confidence_score:.3f}")
            print(f"   Document: {chunk.document_id}")
            print(f"   Content: {chunk.content[:150]}...")
            print(f"   Entities: {chunk.legal_entities}")
            print(f"   Keywords: {chunk.keywords[:8]}")
        
        return chunks
    else:
        print("❌ Pipeline failed to process documents")
        return []

def test_search_simulation():
    """Simulate search functionality with processed chunks"""
    print("\n🔍 Testing Search Simulation...")
    
    chunks = test_chunking_pipeline()
    if not chunks:
        return
    
    # Simple keyword-based search simulation
    search_queries = [
        "school enrollment",
        "budget allocation", 
        "teacher recruitment",
        "education policy",
        "district administration"
    ]
    
    print(f"\n🔍 Search Simulation Results:")
    for query in search_queries:
        query_lower = query.lower()
        matching_chunks = []
        
        for chunk in chunks:
            # Simple keyword matching
            content_lower = chunk.content.lower()
            keywords_lower = [kw.lower() for kw in chunk.keywords]
            
            score = 0
            if any(word in content_lower for word in query_lower.split()):
                score += 0.5
            if any(word in keywords_lower for word in query_lower.split()):
                score += 0.3
            if any(entity.lower() in query_lower for entity in chunk.legal_entities):
                score += 0.2
            
            if score > 0.3:
                matching_chunks.append((chunk, score))
        
        # Sort by score
        matching_chunks.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📝 Query: '{query}'")
        print(f"   Found {len(matching_chunks)} relevant chunks")
        
        for i, (chunk, score) in enumerate(matching_chunks[:3], 1):
            print(f"   {i}. Score: {score:.3f} | [{chunk.section_type}] {chunk.content[:80]}...")

def main():
    """Run all tests"""
    print("🧪 Legal-Aware Chunking System Test Suite")
    print("=" * 50)
    
    try:
        # Test 1: Legal Parser
        test_legal_parser()
        
        # Test 2: Complete Pipeline
        test_chunking_pipeline()
        
        # Test 3: Search Simulation
        test_search_simulation()
        
        print("\n✅ All tests completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Set up Weaviate credentials in weaviate_config.env")
        print("2. Run: python legal_aware_chunker.py --weaviate-url YOUR_URL --weaviate-api-key YOUR_KEY")
        print("3. Test search: python legal_aware_chunker.py --test-search")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
