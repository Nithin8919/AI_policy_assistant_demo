#!/usr/bin/env python3
"""
Simple Data Processor for AP Policy Co-Pilot
Converts extracted text data into searchable format for vector search
"""
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def process_extracted_data():
    """Process extracted data into simple facts for vector search"""
    
    # Load extracted data
    with open('data/extracted/all_extracted_data.json', 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)
    
    # Create output directory
    output_dir = Path('data/vector_search')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize SQLite database
    db_path = output_dir / 'facts.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create facts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            page INTEGER,
            text_id TEXT,
            text_content TEXT,
            source_type TEXT,
            year TEXT,
            extraction_method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Drop existing table if it has wrong schema
    cursor.execute('DROP TABLE IF EXISTS facts')
    cursor.execute('''
        CREATE TABLE facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            page INTEGER,
            text_id TEXT,
            text_content TEXT,
            source_type TEXT,
            year TEXT,
            extraction_method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Clear existing data
    cursor.execute('DELETE FROM facts')
    
    # Process all extracted data
    fact_count = 0
    for doc_name, facts in extracted_data.items():
        for fact in facts:
            # Extract meaningful text (skip empty or very short text)
            text_content = fact.get('text', '').strip()
            if len(text_content) < 10:  # Skip very short text
                continue
            
            # Insert into database
            cursor.execute('''
                INSERT INTO facts (doc_id, page, text_id, text_content, source_type, year, extraction_method)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                fact.get('doc_id', ''),
                fact.get('page', 0),
                fact.get('text_id', ''),
                text_content,
                fact.get('source_type', ''),
                fact.get('year', ''),
                fact.get('extraction_method', '')
            ))
            fact_count += 1
    
    conn.commit()
    conn.close()
    
    print(f"✅ Processed {fact_count} facts into vector search database")
    print(f"📁 Database saved to: {db_path}")
    
    return fact_count

def test_database():
    """Test the created database"""
    db_path = Path('data/vector_search/facts.db')
    if not db_path.exists():
        print("❌ Database not found")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count facts
    cursor.execute('SELECT COUNT(*) FROM facts')
    count = cursor.fetchone()[0]
    print(f"📊 Total facts in database: {count}")
    
    # Show sample facts
    cursor.execute('SELECT doc_id, text_content FROM facts LIMIT 5')
    samples = cursor.fetchall()
    print("\n📝 Sample facts:")
    for i, (doc_id, text) in enumerate(samples, 1):
        print(f"{i}. [{doc_id}] {text[:100]}...")
    
    # Show statistics
    cursor.execute('SELECT source_type, COUNT(*) FROM facts GROUP BY source_type')
    sources = cursor.fetchall()
    print(f"\n📈 Facts by source:")
    for source, count in sources:
        print(f"  {source}: {count}")
    
    conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting simple data processing...")
    fact_count = process_extracted_data()
    
    if fact_count > 0:
        print("\n🧪 Testing database...")
        test_database()
        print("\n✅ Data processing complete! Ready for vector search.")
    else:
        print("❌ No facts processed. Check your extracted data.")
