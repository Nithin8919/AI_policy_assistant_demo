#!/usr/bin/env python3
"""
Table Parsing Efficiency Report
Comprehensive analysis of how well tables are being extracted from PDFs
"""
import json
import os
from collections import Counter

def analyze_table_parsing_efficiency():
    """Analyze the efficiency of table parsing across all documents"""
    
    print("📊 TABLE PARSING EFFICIENCY REPORT")
    print("=" * 60)
    
    # Analyze all extracted files
    extracted_dir = "data/extracted"
    files = [f for f in os.listdir(extracted_dir) if f.endswith('_extracted.json')]
    
    total_facts = 0
    total_table_facts = 0
    extraction_methods = Counter()
    table_methods = Counter()
    document_stats = []
    
    print(f"📄 Analyzing {len(files)} extracted documents...")
    print()
    
    for file_name in files:
        file_path = os.path.join(extracted_dir, file_name)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        facts = data if isinstance(data, list) else data.get('facts', [])
        
        # Count extraction methods
        doc_methods = Counter()
        doc_table_count = 0
        
        for fact in facts:
            if isinstance(fact, dict):
                method = fact.get('extraction_method', 'unknown')
                doc_methods[method] += 1
                extraction_methods[method] += 1
                
                content = fact.get('text', '')
                
                # Check for table indicators
                table_indicators = ['SL.No', 'District', 'Application Id', 'School Code', 'Mandal Name', 'Applicant Name']
                if any(indicator in content for indicator in table_indicators):
                    doc_table_count += 1
                    table_methods[method] += 1
        
        total_facts += len(facts)
        total_table_facts += doc_table_count
        
        # Store document statistics
        doc_stats = {
            'file': file_name,
            'total_facts': len(facts),
            'table_facts': doc_table_count,
            'table_rate': doc_table_count / len(facts) * 100 if len(facts) > 0 else 0,
            'methods': dict(doc_methods)
        }
        document_stats.append(doc_stats)
    
    # Sort documents by table content
    document_stats.sort(key=lambda x: x['table_facts'], reverse=True)
    
    print("📋 DOCUMENT TABLE ANALYSIS:")
    print("-" * 40)
    for i, doc in enumerate(document_stats[:10], 1):  # Top 10 documents
        print(f"{i:2d}. {doc['file'][:30]:<30} | {doc['table_facts']:3d} tables ({doc['table_rate']:4.1f}%)")
    
    print(f"\n📊 OVERALL STATISTICS:")
    print("-" * 25)
    print(f"Total documents analyzed: {len(files)}")
    print(f"Total facts extracted: {total_facts:,}")
    print(f"Total table facts: {total_table_facts:,}")
    print(f"Overall table detection rate: {total_table_facts/total_facts*100:.1f}%")
    
    print(f"\n🔧 EXTRACTION METHODS USED:")
    print("-" * 30)
    for method, count in extraction_methods.most_common():
        percentage = count / total_facts * 100
        print(f"{method:<15}: {count:6,} facts ({percentage:5.1f}%)")
    
    print(f"\n📊 TABLE CONTENT BY EXTRACTION METHOD:")
    print("-" * 40)
    for method, count in table_methods.most_common():
        percentage = count / total_table_facts * 100 if total_table_facts > 0 else 0
        method_total = extraction_methods[method]
        table_rate = count / method_total * 100 if method_total > 0 else 0
        print(f"{method:<15}: {count:4d} tables ({percentage:4.1f}% of tables, {table_rate:4.1f}% of method)")
    
    # Analyze table quality
    print(f"\n🔍 TABLE QUALITY ANALYSIS:")
    print("-" * 30)
    
    # Check a sample of table content for quality
    sample_file = "data/extracted/SCERT_DOCUMENT_1760073238_extracted.json"
    if os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        table_facts = []
        for fact in data:
            if isinstance(fact, dict):
                content = fact.get('text', '')
                if 'SL.No' in content and 'District' in content:
                    table_facts.append(content)
        
        # Analyze table structure
        structured_count = 0
        for content in table_facts[:50]:  # Sample first 50
            if 'SL.No' in content and 'District Name' in content and 'Application Id' in content:
                structured_count += 1
        
        structure_rate = structured_count / len(table_facts[:50]) * 100 if len(table_facts[:50]) > 0 else 0
        
        print(f"Sample table facts analyzed: {len(table_facts[:50])}")
        print(f"Structured table headers: {structured_count}")
        print(f"Structure quality rate: {structure_rate:.1f}%")
    
    # Efficiency assessment
    print(f"\n🎯 EFFICIENCY ASSESSMENT:")
    print("=" * 30)
    
    # Calculate efficiency metrics
    table_detection_rate = total_table_facts / total_facts * 100
    
    # Method effectiveness
    pymupdf_table_rate = table_methods.get('pymupdf_text', 0) / extraction_methods.get('pymupdf_text', 1) * 100
    camelot_table_rate = table_methods.get('camelot_stream', 0) / extraction_methods.get('camelot_stream', 1) * 100
    
    print(f"✅ Table Detection Rate: {table_detection_rate:.1f}%")
    print(f"🔧 PyMuPDF Table Rate: {pymupdf_table_rate:.1f}%")
    print(f"🔧 Camelot Table Rate: {camelot_table_rate:.1f}%")
    
    # Overall efficiency score
    efficiency_score = (table_detection_rate + pymupdf_table_rate + structure_rate) / 3
    
    if efficiency_score >= 80:
        rating = "🟢 EXCELLENT"
        recommendation = "Table parsing is working very well!"
    elif efficiency_score >= 60:
        rating = "🟡 GOOD"
        recommendation = "Table parsing is effective with room for improvement."
    elif efficiency_score >= 40:
        rating = "🟠 FAIR"
        recommendation = "Table parsing needs optimization."
    else:
        rating = "🔴 NEEDS IMPROVEMENT"
        recommendation = "Table parsing requires significant improvements."
    
    print(f"\n🏆 Overall Table Parsing Efficiency: {efficiency_score:.1f}/100")
    print(f"📈 Rating: {rating}")
    print(f"💡 Recommendation: {recommendation}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 15)
    print(f"• {total_table_facts:,} table facts extracted from {len(files)} documents")
    print(f"• PyMuPDF handles {extraction_methods.get('pymupdf_text', 0)/total_facts*100:.1f}% of extractions")
    print(f"• Camelot handles {extraction_methods.get('camelot_stream', 0)/total_facts*100:.1f}% of extractions")
    print(f"• Average {total_table_facts/len(files):.1f} table facts per document")
    print(f"• Table detection rate varies from {min(doc['table_rate'] for doc in document_stats):.1f}% to {max(doc['table_rate'] for doc in document_stats):.1f}%")
    
    return {
        'total_facts': total_facts,
        'total_table_facts': total_table_facts,
        'table_detection_rate': table_detection_rate,
        'efficiency_score': efficiency_score,
        'rating': rating
    }

if __name__ == "__main__":
    results = analyze_table_parsing_efficiency()
    print(f"\n🎉 Analysis complete! Table parsing efficiency: {results['efficiency_score']:.1f}/100")

