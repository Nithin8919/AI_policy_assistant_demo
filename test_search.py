#!/usr/bin/env python3
"""
Quick test of the local vector search system
"""
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pipeline.local_vector_search import LocalVectorSearch

def test_search():
    """Test the search functionality"""
    print("🔍 Testing Local Vector Search System")
    print("=" * 50)
    
    # Initialize search system
    search_system = LocalVectorSearch(db_path="data/vector_search")
    
    # Get statistics
    stats = search_system.get_statistics()
    print(f"📊 Database Statistics:")
    print(f"   Total facts: {stats.get('total_facts', 0):,}")
    print(f"   Unique indicators: {stats.get('unique_indicators', 0)}")
    print(f"   Unique districts: {stats.get('unique_districts', 0)}")
    print(f"   Unique years: {stats.get('unique_years', 0)}")
    print(f"   Vector index size: {stats.get('vector_index_size', 0):,}")
    print(f"   Database type: {stats.get('database', 'Unknown')}")
    print()
    
    # Test queries
    test_queries = [
        "school enrollment statistics",
        "education budget allocation",
        "teacher training programs",
        "dropout rates in Krishna district",
        "literacy rates 2023"
    ]
    
    for query in test_queries:
        print(f"🔍 Query: '{query}'")
        print("-" * 30)
        
        try:
            results = search_system.search(query, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['indicator']} in {result['district']} ({result['year']})")
                    print(f"   Value: {result['value']} {result['unit']}")
                    print(f"   Source: {result['source']}")
                    if 'hybrid_score' in result:
                        print(f"   Score: {result['hybrid_score']:.3f}")
                    print()
            else:
                print("   No results found")
                print()
        
        except Exception as e:
            print(f"   Error: {e}")
            print()
    
    print("✅ Search test completed!")

if __name__ == "__main__":
    test_search()