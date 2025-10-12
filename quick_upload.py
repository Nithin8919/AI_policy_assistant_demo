#!/usr/bin/env python3
"""
Quick Upload to Weaviate
Run this script with your Weaviate URL as an argument
Usage: python quick_upload.py https://your-cluster.weaviate.network
"""
import sys
from legal_aware_chunker import LegalAwareChunker

def main():
    if len(sys.argv) != 2:
        print("Usage: python quick_upload.py <weaviate_url>")
        print("Example: python quick_upload.py https://your-cluster.weaviate.network")
        return
    
    weaviate_url = sys.argv[1]
    api_key = "YXNrUHlTTTZhRWRod0NCR19UM3FacDIxSktIK2htaHFPT29XbnljdVlGYTBPQkpDMXJYODIzRUhuZ2xrPV92MjAw"
    collection_name = "LegalDocuments"
    
    print("🚀 Quick Upload to Weaviate")
    print("=" * 40)
    print(f"URL: {weaviate_url}")
    print(f"Collection: {collection_name}")
    print(f"API Key: {api_key[:20]}...")
    
    try:
        # Initialize chunker with Weaviate
        print("\n🔌 Connecting to Weaviate...")
        chunker = LegalAwareChunker(
            weaviate_url=weaviate_url,
            weaviate_api_key=api_key
        )
        
        if not chunker.weaviate_client:
            print("❌ Failed to connect to Weaviate")
            return
        
        print("✅ Connected to Weaviate successfully!")
        
        # Process documents
        print("\n📄 Processing documents with legal-aware chunking...")
        chunks = chunker.process_documents("data/extracted")
        
        if not chunks:
            print("❌ No chunks created")
            return
        
        print(f"✅ Created {len(chunks)} legal-aware chunks")
        
        # Upload to Weaviate
        print(f"\n💾 Uploading chunks to Weaviate...")
        success = chunker.save_to_weaviate(chunks, collection_name)
        
        if success:
            print("🎉 Successfully uploaded all chunks to Weaviate!")
            
            # Test search
            print("\n🔍 Testing search...")
            results = chunker.search_legal_documents("school enrollment", limit=3)
            
            if results:
                print(f"✅ Search test successful! Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. [{result['section_type']}] {result['content'][:80]}...")
            else:
                print("⚠️  Search test returned no results")
            
            print(f"\n✅ Upload complete!")
            print(f"📊 Collection '{collection_name}' contains {len(chunks)} legal-aware chunks")
            
        else:
            print("❌ Failed to upload chunks to Weaviate")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

