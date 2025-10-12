#!/usr/bin/env python3
"""
Load a sample of processed facts for testing
"""
import sys
import json
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pipeline.local_vector_search import LocalVectorSearch

def load_sample_facts(sample_size=1000):
    """Load a sample of facts for testing"""
    print(f"📥 Loading sample of {sample_size} facts for testing...")
    
    # Load all facts
    facts_file = "data/processed/processed_facts.json"
    try:
        with open(facts_file, 'r', encoding='utf-8') as f:
            all_facts = json.load(f)
    except FileNotFoundError:
        print(f"❌ Facts file not found: {facts_file}")
        return False
    
    print(f"📊 Total facts available: {len(all_facts):,}")
    
    # Take a sample
    sample_facts = all_facts[:sample_size]
    print(f"📦 Loading sample of {len(sample_facts)} facts...")
    
    # Save sample
    sample_file = "data/processed/sample_facts.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_facts, f, indent=2, ensure_ascii=False)
    
    # Initialize search system
    search_system = LocalVectorSearch(db_path="data/vector_search_sample")
    
    # Load sample facts
    success = search_system.load_facts(sample_file)
    
    if success:
        stats = search_system.get_statistics()
        print(f"\n✅ Sample data loaded successfully!")
        print(f"📊 Total facts: {stats.get('total_facts', 0):,}")
        print(f"📈 Unique indicators: {stats.get('unique_indicators', 0)}")
        print(f"🏛️ Unique districts: {stats.get('unique_districts', 0)}")
        print(f"📅 Unique years: {stats.get('unique_years', 0)}")
        print(f"🔍 Vector index size: {stats.get('vector_index_size', 0):,}")
        
        # Test a quick search
        print(f"\n🔍 Testing search with 'budget allocation'...")
        results = search_system.search("budget allocation", limit=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['indicator']} in {result['district']} ({result['year']})")
                print(f"   Value: {result['value']} {result['unit']}")
                print(f"   Source: {result['source']}")
        else:
            print("   No results found")
        
        return True
    else:
        print("❌ Failed to load sample data")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_sample_facts(1000)