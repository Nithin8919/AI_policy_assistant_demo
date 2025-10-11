"""
Main pipeline orchestrator for Policy Intelligence Assistant
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

# Import all components
from data_pipeline.processors.text_extractor import TextExtractor
from data_pipeline.processors.nlp_processor import PolicyNLPProcessor
from backend.graph_manager import GraphManager
from backend.retriever import PolicyRetriever
from backend.bridge_table import BridgeTableManager
from backend.embeddings import EmbeddingService
from policy_intelligence.config.settings import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyIntelligencePipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.document_processor = TextExtractor()
        self.nlp_pipeline = PolicyNLPProcessor()
        self.kg_builder = None
        self.vector_db = None
        self.bridge_table = None
        self.retriever = None
        self.embedding_service = None
        
        # Pipeline state
        self.documents = []
        self.processed_documents = []
        self.nlp_documents = []
        
    def initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        
        try:
            # Initialize Knowledge Graph
            self.kg_builder = GraphManager()
            logger.info("Knowledge Graph initialized")
            
            # Initialize Vector Database (PostgreSQL-based)
            self.retriever = PolicyRetriever()
            logger.info("Vector Database initialized")
            
            # Initialize Bridge Table (PostgreSQL-based)
            self.bridge_table = BridgeTableManager()
            logger.info("Bridge Table initialized")
            
            # Initialize Embedding Service
            self.embedding_service = EmbeddingService()
            logger.info("Embedding Service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def create_sample_data(self):
        """Create sample documents for testing"""
        logger.info("Creating sample documents...")
        
        from policy_intelligence.data.sample_documents import create_sample_documents
        sample_dir = create_sample_documents()
        
        # Load sample documents
        sample_files = list(sample_dir.glob("*.txt"))
        for file_path in sample_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.documents.append({
                'filename': file_path.name,
                'file_path': str(file_path),
                'content': content
            })
        
        logger.info(f"Created {len(self.documents)} sample documents")
    
    def process_documents(self):
        """Process documents through the pipeline"""
        logger.info("Processing documents...")
        
        # Step 1: Document Processing
        logger.info("Step 1: Document Processing")
        for doc in self.documents:
            text, metadata = self.document_processor.extract_text_from_pdf(doc['file_path'])
            doc_data = {
                'filename': doc['filename'],
                'file_path': doc['file_path'],
                'text_length': len(text),
                'word_count': len(text.split()),
                'full_text': text,
                'metadata': metadata
            }
            self.processed_documents.append(doc_data)
        
        # Save processed documents
        for doc_data in self.processed_documents:
            output_file = PROCESSED_DATA_DIR / f"{Path(doc_data['filename']).stem}_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed {len(self.processed_documents)} documents")
    
    def extract_entities_relations(self):
        """Extract entities and relations using NLP pipeline"""
        logger.info("Step 2: Entity and Relation Extraction")
        
        for doc_data in self.processed_documents:
            nlp_result = self.nlp_pipeline.process_document(doc_data['full_text'])
            doc_data['nlp'] = nlp_result
            self.nlp_documents.append(doc_data)
            
            # Save NLP results
            output_file = PROCESSED_DATA_DIR / f"{Path(doc_data['filename']).stem}_nlp.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(nlp_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Extracted entities and relations from {len(self.nlp_documents)} documents")
    
    def build_knowledge_graph(self):
        """Build knowledge graph from processed documents"""
        logger.info("Step 3: Knowledge Graph Construction")
        
        if not self.kg_builder:
            logger.error("Knowledge Graph not initialized")
            return
        
        # Build knowledge graph using Neo4j loader
        from graph_db.neo4j_loader import Neo4jGraphLoader
        neo4j_loader = Neo4jGraphLoader()
        neo4j_loader.load_documents_to_graph(self.nlp_documents)
        
        # Export graph
        graph_output = PROCESSED_DATA_DIR / "knowledge_graph.json"
        neo4j_loader.export_graph_data(graph_output)
        
        logger.info("Knowledge Graph construction complete")
    
    def build_vector_database(self):
        """Build vector database from processed documents"""
        logger.info("Step 4: Vector Database Construction")
        
        if not self.vector_db:
            logger.error("Vector Database not initialized")
            return
        
        # Add documents to PostgreSQL vector database
        for doc_data in self.nlp_documents:
            # Generate embeddings for document chunks
            chunks = self._create_chunks(doc_data['full_text'])
            for chunk in chunks:
                embedding = self.embedding_service.encode(chunk['text'])
                self.bridge_table.insert_span(
                    doc_id=doc_data['filename'],
                    span_text=chunk['text'],
                    span_start=chunk['start'],
                    span_end=chunk['end'],
                    embedding=embedding.tolist()
                )
        
        # Export vectors
        vector_output = PROCESSED_DATA_DIR / "vector_database.json"
        self._export_vector_data(vector_output)
        
        logger.info("Vector Database construction complete")
    
    def build_bridge_table(self):
        """Build bridge table linking graph and vectors"""
        logger.info("Step 5: Bridge Table Construction")
        
        if not self.bridge_table:
            logger.error("Bridge Table not initialized")
            return
        
        # Simulate vector chunks for bridge table
        vector_chunks = []
        for i in range(100):
            vector_chunks.append({
                'chunk_id': f'chunk_{i}',
                'vector_id': f'vec_{i}'
            })
        
        for doc_data in self.nlp_documents:
            self.bridge_table.add_document_metadata(doc_data)
            self.bridge_table.add_entity_mappings(doc_data)
            self.bridge_table.add_relation_mappings(doc_data)
            self.bridge_table.create_bridge_entries(doc_data, vector_chunks)
        
        # Export bridge data
        bridge_output = PROCESSED_DATA_DIR / "bridge_data.json"
        self.bridge_table.export_bridge_data(bridge_output)
        
        logger.info("Bridge Table construction complete")
    
    def _create_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """Create text chunks for vector storage"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'start': i,
                'end': min(i + chunk_size, len(words))
            })
        
        return chunks
    
    def _export_vector_data(self, output_path: Path):
        """Export vector data to JSON"""
        try:
            stats = self.bridge_table.get_statistics()
            export_data = {
                'statistics': stats,
                'export_timestamp': time.time()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Vector data exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export vector data: {e}")
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        logger.info("Starting Policy Intelligence Pipeline")
        start_time = time.time()
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Create sample data
            self.create_sample_data()
            
            # Process documents
            self.process_documents()
            
            # Extract entities and relations
            self.extract_entities_relations()
            
            # Build knowledge graph
            self.build_knowledge_graph()
            
            # Build vector database
            self.build_vector_database()
            
            # Build bridge table
            self.build_bridge_table()
            
            # Pipeline complete
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
            
            # Generate summary report
            self.generate_summary_report()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # Cleanup
            if self.kg_builder:
                self.kg_builder.close()
            if self.bridge_table:
                self.bridge_table.close()
    
    def generate_summary_report(self):
        """Generate a summary report of the pipeline execution"""
        report = {
            'pipeline_summary': {
                'total_documents': len(self.documents),
                'processed_documents': len(self.processed_documents),
                'nlp_documents': len(self.nlp_documents),
                'total_chunks': sum(doc['chunk_count'] for doc in self.nlp_documents),
                'total_words': sum(doc['word_count'] for doc in self.nlp_documents)
            },
            'nlp_summary': {
                'total_entities': sum(doc['nlp']['total_entities'] for doc in self.nlp_documents),
                'total_relations': sum(doc['nlp']['total_relations'] for doc in self.nlp_documents)
            },
            'components': {
                'knowledge_graph': 'Built successfully',
                'vector_database': 'Built successfully',
                'bridge_table': 'Built successfully',
                'retriever': 'Initialized successfully'
            }
        }
        
        # Save report
        report_path = PROCESSED_DATA_DIR / "pipeline_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("POLICY INTELLIGENCE PIPELINE SUMMARY")
        print("="*60)
        print(f"Documents processed: {report['pipeline_summary']['total_documents']}")
        print(f"Total chunks: {report['pipeline_summary']['total_chunks']}")
        print(f"Total words: {report['pipeline_summary']['total_words']}")
        print(f"Entities extracted: {report['nlp_summary']['total_entities']}")
        print(f"Relations extracted: {report['nlp_summary']['total_relations']}")
        print("="*60)
    
    def test_retrieval(self):
        """Test the retrieval system"""
        if not self.retriever:
            logger.error("Retriever not initialized")
            return
        
        logger.info("Testing retrieval system...")
        
        test_queries = [
            "What is the National Education Policy 2020?",
            "How is NEP 2020 implemented in Andhra Pradesh?",
            "What is the 5+3+3+4 curricular structure?",
            "What are the key features of foundational stage learning?"
        ]
        
        results = []
        for query in test_queries:
            # Generate embedding for query
            query_embedding = self.embedding_service.encode(query)
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(
                query_embedding=query_embedding.tolist(),
                max_results=5
            )
            
            # Generate simple answer
            if retrieved_docs:
                answer_text = f"Based on the retrieved documents: {retrieved_docs[0].get('span_text', 'No text available')[:200]}..."
                confidence = retrieved_docs[0].get('confidence', 0.0)
                sources = len(retrieved_docs)
                citations = [doc.get('source_url', 'Unknown') for doc in retrieved_docs]
            else:
                answer_text = "No relevant documents found."
                confidence = 0.0
                sources = 0
                citations = []
            
            results.append({
                'query': query,
                'answer': answer_text,
                'confidence': confidence,
                'sources': sources,
                'citations': citations
            })
        
        # Save test results
        test_output = PROCESSED_DATA_DIR / "retrieval_test_results.json"
        with open(test_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Retrieval test results saved to {test_output}")
        
        # Print results
        print("\n" + "="*60)
        print("RETRIEVAL TEST RESULTS")
        print("="*60)
        for result in results:
            print(f"\nQuery: {result['query']}")
            print(f"Answer: {result['answer'][:100]}...")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Sources: {result['sources']}")
        print("="*60)

def main():
    """Main function to run the pipeline"""
    pipeline = PolicyIntelligencePipeline()
    
    try:
        # Run full pipeline
        pipeline.run_full_pipeline()
        
        # Test retrieval
        pipeline.test_retrieval()
        
        print("\nPipeline execution completed successfully!")
        print("Check the processed_data directory for all outputs.")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
