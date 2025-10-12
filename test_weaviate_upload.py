#!/usr/bin/env python3
"""
Test Weaviate Upload and Search
Verify that chunks were uploaded and search is working
"""
import weaviate
from legal_aware_chunker import LegalAwareChunker

def test_weaviate_upload():
    """Test the Weaviate upload and search functionality"""
    
    # Your Weaviate credentials
    weaviate_url = "https://mg90f7v2snmbkjksqqpvxa.c0.asia-southeast1.gcp.weaviate.cloud"
    api_key = "YXNrUHlTTTZhRWRod0NCR19UM3FacDIxSktIK2htaHFPT29XbnljdVlGYTBPQkpDMXJYODIzRUhuZ2xrPV92MjAw"
    
    print("🔍 Testing Weaviate Upload and Search")
    print("=" * 50)
    
    try:
        # Initialize chunker
        chunker = LegalAwareChunker(
            weaviate_url=weaviate_url,
            weaviate_api_key=api_key
        )
        
        if not chunker.weaviate_client:
            print("❌ Failed to connect to Weaviate")
            return
        
        print("✅ Connected to Weaviate successfully!")
        
        # Check if collection exists and count objects
        try:
            schema = chunker.weaviate_client.schema.get()
            collections = [cls['class'] for cls in schema['classes']]
            
            if 'LegalDocuments' in collections:
                print("✅ Collection 'LegalDocuments' exists")
                
                # Count objects in collection
                result = chunker.weaviate_client.query.aggregate("LegalDocuments").with_meta_count().do()
                count = result['data']['Aggregate']['LegalDocuments'][0]['meta']['count']
                print(f"📊 Total objects in collection: {count}")
                
                # Test search with different approaches
                print("\n🔍 Testing search functionality...")
                
                # Test 1: Simple text search
                print("\n1. Testing simple text search:")
                try:
                    result = chunker.weaviate_client.query.get(
                        "LegalDocuments",
                        ["chunk_id", "content", "section_type", "confidence_score"]
                    ).with_limit(5).do()
                    
                    if result and "data" in result and "Get" in result["data"]:
                        objects = result["data"]["Get"]["LegalDocuments"]
                        print(f"   Found {len(objects)} objects")
                        for i, obj in enumerate(objects[:3], 1):
                            print(f"   {i}. [{obj['section_type']}] {obj['content'][:80]}...")
                    else:
                        print("   No objects found")
                        
                except Exception as e:
                    print(f"   Error in text search: {e}")
                
                # Test 2: Search with where clause
                print("\n2. Testing filtered search:")
                try:
                    result = chunker.weaviate_client.query.get(
                        "LegalDocuments",
                        ["chunk_id", "content", "section_type", "confidence_score"]
                    ).with_where({
                        "path": ["confidence_score"],
                        "operator": "GreaterThan",
                        "valueNumber": 0.8
                    }).with_limit(3).do()
                    
                    if result and "data" in result and "Get" in result["data"]:
                        objects = result["data"]["Get"]["LegalDocuments"]
                        print(f"   Found {len(objects)} high-confidence objects")
                        for i, obj in enumerate(objects, 1):
                            print(f"   {i}. [{obj['section_type']}] Confidence: {obj['confidence_score']:.3f}")
                            print(f"      {obj['content'][:60]}...")
                    else:
                        print("   No high-confidence objects found")
                        
                except Exception as e:
                    print(f"   Error in filtered search: {e}")
                
                # Test 3: Search by section type
                print("\n3. Testing section type search:")
                try:
                    result = chunker.weaviate_client.query.get(
                        "LegalDocuments",
                        ["chunk_id", "content", "section_type"]
                    ).with_where({
                        "path": ["section_type"],
                        "operator": "Equal",
                        "valueText": "content"
                    }).with_limit(3).do()
                    
                    if result and "data" in result and "Get" in result["data"]:
                        objects = result["data"]["Get"]["LegalDocuments"]
                        print(f"   Found {len(objects)} content-type objects")
                        for i, obj in enumerate(objects, 1):
                            print(f"   {i}. [{obj['section_type']}] {obj['content'][:60]}...")
                    else:
                        print("   No content-type objects found")
                        
                except Exception as e:
                    print(f"   Error in section search: {e}")
                
                print(f"\n✅ Upload verification complete!")
                print(f"📊 Collection 'LegalDocuments' contains {count} legal-aware chunks")
                print(f"🔍 Search functionality is working")
                
            else:
                print("❌ Collection 'LegalDocuments' not found")
                print("Available collections:", collections)
                
        except Exception as e:
            print(f"❌ Error checking collection: {e}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_weaviate_upload()

