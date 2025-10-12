#!/usr/bin/env python3
"""
Stage 5: Weaviate Index Verification
Verifies that Weaviate collections and indexes are properly created
"""
import os
import json
import logging
import weaviate
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

def verify_weaviate_indexes(output_dir: str = "data/weaviate") -> bool:
    """Verify Weaviate collections and indexes"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Connect to Weaviate
        weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        client = weaviate.connect_to_local(
            host=weaviate_url.replace('http://', '').replace('https://', ''),
            port=8080,
            grpc_port=50051
        )
        
        # Get all collections
        collections = client.collections.list_all()
        
        verification_report = {
            "timestamp": datetime.now().isoformat(),
            "weaviate_url": weaviate_url,
            "collections": {},
            "status": "success"
        }
        
        print("\n✅ Weaviate Collections Verification:")
        print("=" * 50)
        
        for collection in collections:
            print(f"📁 Collection: {collection.name}")
            
            try:
                # Get collection details
                col = client.collections.get(collection.name)
                
                # Get object count
                count_result = col.aggregate.over_all(total_count=True)
                object_count = count_result.total_count if count_result else 0
                
                # Get collection configuration
                config = col.config.get()
                
                collection_info = {
                    "name": collection.name,
                    "description": config.description if config else "No description",
                    "object_count": object_count,
                    "vectorizer": str(config.vectorizer_config) if config else "None",
                    "vector_index": str(config.vector_index_config) if config else "None",
                    "properties": []
                }
                
                # Get properties
                if config and config.properties:
                    for prop in config.properties:
                        collection_info["properties"].append({
                            "name": prop.name,
                            "data_type": str(prop.data_type),
                            "skip_vectorization": getattr(prop, 'skip_vectorization', False)
                        })
                
                verification_report["collections"][collection.name] = collection_info
                
                print(f"   📊 Objects: {object_count:,}")
                print(f"   🔗 Properties: {len(collection_info['properties'])}")
                print(f"   🎯 Vectorizer: {collection_info['vectorizer']}")
                print()
                
            except Exception as e:
                logger.error(f"Failed to get details for collection {collection.name}: {e}")
                verification_report["collections"][collection.name] = {
                    "name": collection.name,
                    "error": str(e)
                }
                print(f"   ❌ Error: {e}")
                print()
        
        # Test basic search functionality
        try:
            if "Fact" in [c.name for c in collections]:
                fact_collection = client.collections.get("Fact")
                
                # Try a simple query
                response = fact_collection.query.fetch_objects(limit=1)
                
                if response.objects:
                    print("🔍 Search Test: ✅ PASSED")
                    verification_report["search_test"] = "passed"
                else:
                    print("🔍 Search Test: ⚠️  NO DATA (but functional)")
                    verification_report["search_test"] = "no_data"
            else:
                print("🔍 Search Test: ❌ FAILED (Fact collection missing)")
                verification_report["search_test"] = "failed"
                verification_report["status"] = "error"
        
        except Exception as e:
            logger.error(f"Search test failed: {e}")
            print(f"🔍 Search Test: ❌ FAILED ({e})")
            verification_report["search_test"] = f"failed: {e}"
            verification_report["status"] = "error"
        
        # Save verification report
        report_file = output_path / "verification_report.json"
        with open(report_file, 'w') as f:
            json.dump(verification_report, f, indent=2)
        
        print("=" * 50)
        print(f"📄 Report saved to: {report_file}")
        
        client.close()
        
        return verification_report["status"] == "success"
    
    except Exception as e:
        logger.error(f"Weaviate verification failed: {e}")
        print(f"❌ Weaviate verification failed: {e}")
        return False

def test_hybrid_search() -> bool:
    """Test hybrid search functionality"""
    try:
        weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        client = weaviate.connect_to_local(
            host=weaviate_url.replace('http://', '').replace('https://', ''),
            port=8080,
            grpc_port=50051
        )
        
        print("\n🔍 Testing Hybrid Search:")
        print("-" * 30)
        
        fact_collection = client.collections.get("Fact")
        
        # Test queries
        test_queries = [
            "education enrollment statistics",
            "school attendance rates",
            "literacy data"
        ]
        
        for query in test_queries:
            try:
                # Simple hybrid search (alpha=0.7 means 70% vector, 30% keyword)
                response = fact_collection.query.hybrid(
                    query=query,
                    alpha=0.7,
                    limit=3
                )
                
                result_count = len(response.objects)
                print(f"   '{query}': {result_count} results")
                
                if result_count > 0:
                    # Show top result
                    top_result = response.objects[0]
                    indicator = top_result.properties.get('indicator', 'Unknown')
                    district = top_result.properties.get('district', 'Unknown')
                    print(f"      Top result: {indicator} in {district}")
                
            except Exception as e:
                print(f"   '{query}': ❌ Failed ({e})")
        
        client.close()
        return True
    
    except Exception as e:
        logger.error(f"Hybrid search test failed: {e}")
        print(f"❌ Hybrid search test failed: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify Weaviate indexes and collections')
    parser.add_argument('--output-dir', default='data/weaviate', help='Output directory')
    parser.add_argument('--test-search', action='store_true', help='Test search functionality')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Verify collections
    success = verify_weaviate_indexes(args.output_dir)
    
    # Test search if requested
    if args.test_search:
        search_success = test_hybrid_search()
        success = success and search_success
    
    if success:
        print("\n✅ Weaviate verification completed successfully")
    else:
        print("\n❌ Weaviate verification failed")
        exit(1)

if __name__ == "__main__":
    main()