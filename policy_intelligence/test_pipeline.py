"""
Test script for Policy Intelligence Assistant
"""
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_sample_data_creation():
    """Test sample data creation"""
    print("Testing sample data creation...")
    
    try:
        from policy_intelligence.data.sample_documents import create_sample_documents
        sample_dir = create_sample_documents()
        
        # Check if files were created
        sample_files = list(sample_dir.glob("*.txt"))
        metadata_file = sample_dir / "metadata.json"
        
        if len(sample_files) >= 3 and metadata_file.exists():
            print("✅ Sample data creation test passed")
            return True
        else:
            print("❌ Sample data creation test failed")
            return False
            
    except Exception as e:
        print(f"❌ Sample data creation test failed: {e}")
        return False

def test_document_processing():
    """Test document processing"""
    print("Testing document processing...")
    
    try:
        from data_pipeline.processors.text_extractor import TextExtractor
        
        processor = TextExtractor()
        
        # Test with sample data
        sample_dir = Path(__file__).parent / "data" / "sample"
        if not sample_dir.exists():
            print("❌ Sample data not found")
            return False
        
        sample_files = list(sample_dir.glob("*.txt"))
        if not sample_files:
            print("❌ No sample files found")
            return False
        
        # Process first sample file
        text, metadata = processor.extract_text_from_pdf(str(sample_files[0]))
        doc_data = {
            'filename': sample_files[0].name,
            'text_length': len(text),
            'word_count': len(text.split()),
            'full_text': text,
            'metadata': metadata
        }
        
        if doc_data and 'full_text' in doc_data and len(doc_data['full_text']) > 0:
            print("✅ Document processing test passed")
            return True
        else:
            print("❌ Document processing test failed")
            return False
            
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False

def test_nlp_pipeline():
    """Test NLP pipeline"""
    print("Testing NLP pipeline...")
    
    try:
        from data_pipeline.processors.nlp_processor import PolicyNLPProcessor
        
        pipeline = PolicyNLPProcessor()
        
        # Test text
        test_text = "The National Education Policy 2020 introduces a 5+3+3+4 curricular structure. The Government of Andhra Pradesh issued GO MS No.45 to implement this policy."
        
        result = pipeline.process_text(test_text)
        
        if result and 'entities' in result and 'relations' in result:
            print(f"✅ NLP pipeline test passed - Found {len(result['entities'])} entities and {len(result['relations'])} relations")
            return True
        else:
            print("❌ NLP pipeline test failed")
            return False
            
    except Exception as e:
        print(f"❌ NLP pipeline test failed: {e}")
        return False

def test_retrieval_system():
    """Test retrieval system"""
    print("Testing retrieval system...")
    
    try:
        from backend.retriever import PolicyRetriever
        
        # Test retrieval with mock embedding
        retriever = PolicyRetriever()
        
        # Mock query embedding
        mock_embedding = [0.1] * 384  # Mock 384-dimensional embedding
        
        # Test query
        results = retriever.retrieve(
            query_embedding=mock_embedding,
            max_results=5
        )
        
        if results is not None:
            print("✅ Retrieval system test passed")
            return True
        else:
            print("❌ Retrieval system test failed")
            return False
            
    except Exception as e:
        print(f"❌ Retrieval system test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Running Policy Intelligence Assistant Tests")
    print("=" * 50)
    
    tests = [
        test_sample_data_creation,
        test_document_processing,
        test_nlp_pipeline,
        test_retrieval_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

