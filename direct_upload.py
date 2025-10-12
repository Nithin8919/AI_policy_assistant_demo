#!/usr/bin/env python3
"""
Direct Upload to Weaviate
Non-interactive script to upload legal chunks to Weaviate
"""
import os
import sys
from legal_aware_chunker import LegalAwareChunker

def upload_chunks_to_weaviate(weaviate_url, api_key, collection_name="LegalDocuments"):
    """Upload chunks to Weaviate with provided credentials"""
    
    print("🚀 Uploading Legal Chunks to Weaviate")
    print("=" * 50)
    print(f"URL: {weaviate_url}")
    print(f"Collection: {collection_name}")
    print(f"API Key: {'*' * len(api_key)}")
    
    try:
        # Initialize chunker with Weaviate
        print("\n🔌 Connecting to Weaviate...")
        chunker = LegalAwareChunker(
            weaviate_url=weaviate_url,
            weaviate_api_key=api_key
        )
        
        if not chunker.weaviate_client:
            print("❌ Failed to connect to Weaviate")
            return False
        
        print("✅ Connected to Weaviate successfully!")
        
        # Process documents (this will create chunks with embeddings)
        print("\n📄 Processing documents with legal-aware chunking...")
        chunks = chunker.process_documents("data/extracted")
        
        if not chunks:
            print("❌ No chunks created")
            return False
        
        print(f"✅ Created {len(chunks)} legal-aware chunks")
        
        # Upload to Weaviate
        print(f"\n💾 Uploading chunks to Weaviate collection '{collection_name}'...")
        success = chunker.save_to_weaviate(chunks, collection_name)
        
        if success:
            print("🎉 Successfully uploaded all chunks to Weaviate!")
            
            # Test search
            print("\n🔍 Testing search functionality...")
            test_queries = [
                "school enrollment policy",
                "budget allocation for education", 
                "teacher recruitment guidelines",
                "district administration procedures"
            ]
            
            for query in test_queries:
                print(f"\n📝 Query: '{query}'")
                results = chunker.search_legal_documents(
                    query, 
                    limit=3,
                    min_confidence=0.3
                )
                
                if results:
                    print(f"   Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"   {i}. [{result['section_type']}] Confidence: {result['confidence_score']:.3f}")
                        print(f"      {result['content'][:100]}...")
                        print(f"      Document: {result['document_id']}")
                else:
                    print("   No results found")
            
            print(f"\n✅ Upload complete!")
            print(f"📊 Collection '{collection_name}' now contains {len(chunks)} legal-aware chunks")
            print(f"🔍 You can now search using the Weaviate API or GraphQL interface")
            
            return True
            
        else:
            print("❌ Failed to upload chunks to Weaviate")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Your Weaviate credentials
    WEAVIATE_URL = "https://your-cluster-url.weaviate.network"  # Replace with your actual URL
    API_KEY = "YXNrUHlTTTZhRWRod0NCR19UM3FacDIxSktIK2htaHFPT29XbnljdVlGYTBPQkpDMXJYODIzRUhuZ2xrPV92MjAw"
    COLLECTION_NAME = "LegalDocuments"
    
    print("⚠️  Please update the WEAVIATE_URL variable with your actual cluster URL")
    print("   Current URL:", WEAVIATE_URL)
    print("   API Key:", API_KEY[:20] + "...")
    
    # Uncomment the line below after updating the URL
    # success = upload_chunks_to_weaviate(WEAVIATE_URL, API_KEY, COLLECTION_NAME)
    
    print("\n📋 To upload your chunks:")
    print("1. Update the WEAVIATE_URL variable in this script with your cluster URL")
    print("2. Uncomment the upload line")
    print("3. Run: python direct_upload.py")

