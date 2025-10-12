#!/usr/bin/env python3
"""
Simple Vector Search Test for AP Policy Co-Pilot
Tests search functionality with the processed data
"""
import sqlite3
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)

class SimpleVectorSearch:
    """Simple vector search using sentence transformers"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.facts = []
        self.db_path = 'data/vector_search/facts.db'
    
    def load_data(self, limit: int = 1000):
        """Load facts from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f'SELECT id, doc_id, text_content, source_type, year FROM facts LIMIT {limit}')
        rows = cursor.fetchall()
        
        self.facts = []
        for row in rows:
            self.facts.append({
                'id': row[0],
                'doc_id': row[1],
                'text': row[2],
                'source_type': row[3],
                'year': row[4]
            })
        
        conn.close()
        print(f"📥 Loaded {len(self.facts)} facts")
    
    def create_embeddings(self):
        """Create embeddings for all facts"""
        if not self.facts:
            print("❌ No facts loaded")
            return
        
        texts = [fact['text'] for fact in self.facts]
        print("🔄 Creating embeddings...")
        self.embeddings = self.model.encode(texts)
        print(f"✅ Created embeddings: {self.embeddings.shape}")
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar facts"""
        if self.embeddings is None:
            print("❌ No embeddings created")
            return []
        
        # Create query embedding
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit]
        
        results = []
        for idx in top_indices:
            fact = self.facts[idx]
            results.append({
                'score': float(similarities[idx]),
                'text': fact['text'],
                'doc_id': fact['doc_id'],
                'source_type': fact['source_type'],
                'year': fact['year']
            })
        
        return results

def main():
    """Test the vector search system"""
    print("🚀 Starting Simple Vector Search Test...")
    
    # Initialize search
    search = SimpleVectorSearch()
    
    # Load data
    search.load_data(limit=1000)
    
    if not search.facts:
        print("❌ No facts found in database")
        return
    
    # Create embeddings
    search.create_embeddings()
    
    # Test searches
    test_queries = [
        "school enrollment",
        "budget allocation",
        "teacher recruitment",
        "district education",
        "student performance"
    ]
    
    print("\n🔍 Testing search queries:")
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        results = search.search(query, limit=3)
        
        if results:
            print(f"   Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. Score: {result['score']:.3f}")
                print(f"      Text: {result['text'][:80]}...")
                print(f"      Source: {result['source_type']} - {result['doc_id']}")
        else:
            print("   No results found")
    
    print("\n✅ Vector search test complete!")

if __name__ == "__main__":
    main()

