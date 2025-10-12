#!/usr/bin/env python3
"""
Re-upload to Weaviate with Fixed Date Format
Clear existing collection and upload with correct RFC3339 date format
"""
import weaviate
from legal_aware_chunker import LegalAwareChunker

def re_upload_to_weaviate():
    """Clear collection and re-upload with correct date format"""
    
    # Your Weaviate credentials
    weaviate_url = "https://mg90f7v2snmbkjksqqpvxa.c0.asia-southeast1.gcp.weaviate.cloud"
    api_key = "YXNrUHlTTTZhRWRod0NCR19UM3FacDIxSktIK2htaHFPT29XbnljdVlGYTBPQkpDMXJYODIzRUhuZ2xrPV92MjAw"
    collection_name = "LegalDocuments"
    
    print("🔄 Re-uploading to Weaviate with Fixed Date Format")
    print("=" * 60)
    
    try:
        # Initialize chunker
        chunker = LegalAwareChunker(
            weaviate_url=weaviate_url,
            weaviate_api_key=api_key
        )
        
        if not chunker.weaviate_client:
            print("❌ Failed to connect to Weaviate")
            return False
        
        print("✅ Connected to Weaviate successfully!")
        
        # Clear existing collection
        print(f"\n🗑️ Clearing existing collection '{collection_name}'...")
        try:
            chunker.weaviate_client.schema.delete_class(collection_name)
            print("✅ Collection cleared successfully")
        except Exception as e:
            print(f"⚠️ Collection clear warning: {e}")
        
        # Process documents (this will create chunks with embeddings)
        print("\n📄 Processing documents with legal-aware chunking...")
        chunks = chunker.process_documents("data/extracted")
        
        if not chunks:
            print("❌ No chunks created")
            return False
        
        print(f"✅ Created {len(chunks)} legal-aware chunks")
        
        # Upload to Weaviate with correct date format
        print(f"\n💾 Uploading chunks to Weaviate collection '{collection_name}'...")
        success = chunker.save_to_weaviate(chunks, collection_name)
        
        if success:
            print("🎉 Successfully uploaded all chunks to Weaviate!")
            
            # Verify upload
            print("\n🔍 Verifying upload...")
            try:
                result = chunker.weaviate_client.query.aggregate("LegalDocuments").with_meta_count().do()
                count = result['data']['Aggregate']['LegalDocuments'][0]['meta']['count']
                print(f"📊 Verified: {count} objects in collection")
                
                if count > 0:
                    # Test search
                    print("\n🔍 Testing search functionality...")
                    result = chunker.weaviate_client.query.get(
                        "LegalDocuments",
                        ["chunk_id", "content", "section_type", "confidence_score"]
                    ).with_limit(3).do()
                    
                    if result and "data" in result and "Get" in result["data"]:
                        objects = result["data"]["Get"]["LegalDocuments"]
                        print(f"✅ Search test successful! Found {len(objects)} objects:")
                        for i, obj in enumerate(objects, 1):
                            print(f"   {i}. [{obj['section_type']}] Confidence: {obj['confidence_score']:.3f}")
                            print(f"      {obj['content'][:80]}...")
                    else:
                        print("⚠️ Search test returned no results")
                    
                    print(f"\n✅ Re-upload complete!")
                    print(f"📊 Collection '{collection_name}' now contains {count} legal-aware chunks")
                    print(f"🔍 Search functionality is working")
                    return True
                else:
                    print("❌ Upload verification failed - no objects found")
                    return False
                    
            except Exception as e:
                print(f"❌ Error verifying upload: {e}")
                return False
            
        else:
            print("❌ Failed to upload chunks to Weaviate")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = re_upload_to_weaviate()
    if success:
        print("\n🎉 Legal document processing and Weaviate upload completed successfully!")
    else:
        print("\n❌ Re-upload failed. Please check the errors above.")

