#!/usr/bin/env python3
"""
Upload Legal Chunks to Weaviate
Simple script to upload processed chunks to your Weaviate instance
"""
import os
import sys
from legal_aware_chunker import LegalAwareChunker

def upload_to_weaviate():
    """Upload chunks to Weaviate with user-provided credentials"""
    
    print("🚀 Upload Legal Chunks to Weaviate")
    print("=" * 50)
    
    # Get credentials from user
    print("Please provide your Weaviate credentials:")
    weaviate_url = input("Weaviate URL (e.g., https://your-cluster.weaviate.network): ").strip()
    
    if not weaviate_url:
        print("❌ Weaviate URL is required")
        return False
    
    weaviate_api_key = input("Weaviate API Key: ").strip()
    
    if not weaviate_api_key:
        print("❌ Weaviate API Key is required")
        return False
    
    collection_name = input("Collection name (default: LegalDocuments): ").strip() or "LegalDocuments"
    
    print(f"\n📋 Configuration:")
    print(f"   URL: {weaviate_url}")
    print(f"   Collection: {collection_name}")
    print(f"   API Key: {'*' * len(weaviate_api_key)}")
    
    confirm = input("\nProceed with upload? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Upload cancelled")
        return False
    
    try:
        # Initialize chunker with Weaviate
        print("\n🔌 Connecting to Weaviate...")
        chunker = LegalAwareChunker(
            weaviate_url=weaviate_url,
            weaviate_api_key=weaviate_api_key
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
    success = upload_to_weaviate()
    if success:
        print("\n🎉 Legal document processing and Weaviate upload completed successfully!")
    else:
        print("\n❌ Upload failed. Please check your credentials and try again.")
