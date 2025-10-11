#!/usr/bin/env python3
"""
Policy Intelligence Assistant - CLI Interface
"""
import argparse
import sys
from pathlib import Path
import json
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from policy_intelligence.main_pipeline import PolicyIntelligencePipeline
from policy_intelligence.config.settings import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    """Run the complete pipeline"""
    print("🚀 Starting Policy Intelligence Assistant Pipeline")
    print("=" * 60)
    
    pipeline = PolicyIntelligencePipeline()
    
    try:
        # Run full pipeline
        pipeline.run_full_pipeline()
        
        # Test retrieval
        pipeline.test_retrieval()
        
        print("\n✅ Pipeline execution completed successfully!")
        print("📁 Check the processed_data directory for all outputs.")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        return 1
    
    return 0

def interactive_qa():
    """Interactive Q&A mode"""
    print("🤖 Policy Intelligence Assistant - Interactive Q&A")
    print("=" * 60)
    print("Ask questions about Andhra Pradesh education policies.")
    print("Type 'quit' to exit.\n")
    
    # Sample questions for reference
    sample_questions = [
        "What is the National Education Policy 2020?",
        "How is NEP 2020 implemented in Andhra Pradesh?",
        "What is the 5+3+3+4 curricular structure?",
        "What are the key features of foundational stage learning?",
        "What government orders relate to education policy?"
    ]
    
    print("Sample questions:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")
    print()
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the Policy Intelligence Assistant!")
                break
            
            if not query:
                print("Please enter a question.")
                continue
            
            # Simulate answer generation
            print("🔍 Processing your question...")
            
            # Simple keyword matching for demo
            if any(keyword in query.lower() for keyword in ['nep', 'national education policy', '2020']):
                answer = "The National Education Policy 2020 is a comprehensive framework for educational reform in India. It introduces a 5+3+3+4 curricular structure and emphasizes foundational learning, multilingualism, and experiential learning across different stages of education."
                confidence = 0.92
                sources = ["NEP 2020 Document", "AP GO MS No.45", "Court Judgment Sample"]
            elif any(keyword in query.lower() for keyword in ['5+3+3+4', 'curricular structure', 'structure']):
                answer = "The 5+3+3+4 curricular structure consists of four stages: Foundational Stage (ages 3-8) with play-based learning, Preparatory Stage (ages 8-11) with activity-based learning, Middle Stage (ages 11-14) with experiential learning, and Secondary Stage (ages 14-18) with multidisciplinary education."
                confidence = 0.95
                sources = ["NEP 2020 Document", "AP GO MS No.45"]
            elif any(keyword in query.lower() for keyword in ['foundational', 'foundation', 'stage']):
                answer = "The Foundational Stage (ages 3-8) focuses on play-based learning, early childhood care and education, and developing basic literacy and numeracy skills. It emphasizes flexible, multilevel learning approaches."
                confidence = 0.88
                sources = ["NEP 2020 Document"]
            elif any(keyword in query.lower() for keyword in ['andhra pradesh', 'ap', 'implementation', 'government']):
                answer = "Andhra Pradesh implements education policies through government orders issued by the School Education Department. The state has adopted the NEP 2020 structure and is working on foundational literacy and numeracy programs."
                confidence = 0.90
                sources = ["AP GO MS No.45", "Court Judgment Sample"]
            else:
                answer = "I found some relevant information in the policy documents, but I need more specific details to provide a comprehensive answer. Could you please rephrase your question or ask about a specific aspect of the education policy?"
                confidence = 0.65
                sources = ["General Policy Documents"]
            
            print(f"\n💡 Answer: {answer}")
            print(f"\n📊 Confidence: {confidence:.1%}")
            print(f"\n📚 Sources:")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source}")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing question: {e}")

def show_status():
    """Show pipeline status and statistics"""
    print("📊 Policy Intelligence Assistant - Status")
    print("=" * 50)
    
    processed_dir = PROCESSED_DATA_DIR
    
    if not processed_dir.exists():
        print("❌ No processed data found. Run the pipeline first.")
        return
    
    # Count files
    processed_files = list(processed_dir.glob("*_processed.json"))
    nlp_files = list(processed_dir.glob("*_nlp.json"))
    
    print(f"📄 Processed documents: {len(processed_files)}")
    print(f"🧠 NLP processed documents: {len(nlp_files)}")
    
    # Check for key outputs
    outputs = {
        "Knowledge Graph": processed_dir / "knowledge_graph.json",
        "Vector Database": processed_dir / "vector_database.json", 
        "Bridge Table": processed_dir / "bridge_table.db",
        "Pipeline Report": processed_dir / "pipeline_report.json"
    }
    
    print("\n📁 Output Files:")
    for name, path in outputs.items():
        status = "✅" if path.exists() else "❌"
        print(f"  {status} {name}: {path.name}")
    
    # Show statistics if available
    report_file = processed_dir / "pipeline_report.json"
    if report_file.exists():
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            print("\n📈 Statistics:")
            summary = report.get('pipeline_summary', {})
            print(f"  • Total documents: {summary.get('total_documents', 0)}")
            print(f"  • Total chunks: {summary.get('total_chunks', 0)}")
            print(f"  • Total words: {summary.get('total_words', 0)}")
            
            nlp_summary = report.get('nlp_summary', {})
            print(f"  • Total entities: {nlp_summary.get('total_entities', 0)}")
            print(f"  • Total relations: {nlp_summary.get('total_relations', 0)}")
            
        except Exception as e:
            print(f"❌ Error reading report: {e}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Policy Intelligence Assistant - Andhra Pradesh Education Policy Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py run                    # Run complete pipeline
  python cli.py qa                    # Interactive Q&A mode
  python cli.py status                # Show pipeline status
        """
    )
    
    parser.add_argument(
        'command',
        choices=['run', 'qa', 'status'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.command == 'run':
        return run_pipeline()
    elif args.command == 'qa':
        interactive_qa()
        return 0
    elif args.command == 'status':
        show_status()
        return 0

if __name__ == "__main__":
    sys.exit(main())

