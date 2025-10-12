#!/usr/bin/env python3
"""
Database Setup Utility - Weaviate + Neo4j
Sets up Weaviate vector database and Neo4j knowledge graph
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import subprocess

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.init import Auth

# Neo4j (keep as-is)
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: neo4j package not installed")

logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Setup Weaviate and Neo4j databases"""
    
    def __init__(self):
        # Weaviate configuration
        self.weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        self.weaviate_timeout = int(os.getenv('WEAVIATE_TIMEOUT', '30'))
        
        # Neo4j configuration (unchanged)
        self.neo4j_config = {
            'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            'user': os.getenv('NEO4J_USER', 'neo4j'),
            'password': os.getenv('NEO4J_PASSWORD', 'password')
        }
    
    def setup_weaviate(self) -> bool:
        """Setup Weaviate schema and collections"""
        logger.info("Setting up Weaviate database...")
        
        try:
            # Connect to Weaviate
            client = weaviate.connect_to_local(
                host=self.weaviate_url.replace('http://', '').replace('https://', ''),
                port=8080,
                grpc_port=50051
            )
            
            logger.info("Connected to Weaviate")
            
            # Check if collections already exist
            existing_collections = [c.name for c in client.collections.list_all()]
            
            # Create Fact collection
            if "Fact" not in existing_collections:
                client.collections.create(
                    name="Fact",
                    description="Educational policy facts with metrics and metadata",
                    properties=[
                        Property(name="fact_id", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="indicator", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="category", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="district", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="year", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="value", data_type=DataType.NUMBER),
                        Property(name="unit", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="source", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="page_ref", data_type=DataType.INT),
                        Property(name="confidence", data_type=DataType.NUMBER),
                        Property(name="table_id", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="pdf_name", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="span_text", data_type=DataType.TEXT),  # This gets vectorized
                        Property(name="created_at", data_type=DataType.DATE),
                    ],
                    vectorizer_config=Configure.Vectorizer.none(),  # We provide embeddings
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric="cosine",
                        ef_construction=128,
                        ef=64,
                        max_connections=32
                    )
                )
                logger.info("Created Fact collection")
            else:
                logger.info("Fact collection already exists")
            
            # Create Document collection
            if "Document" not in existing_collections:
                client.collections.create(
                    name="Document",
                    description="Source documents (PDFs, GOs, Reports)",
                    properties=[
                        Property(name="doc_id", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="filename", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="source_type", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="year", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="total_pages", data_type=DataType.INT),
                        Property(name="extraction_method", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="checksum", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="file_path", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="created_at", data_type=DataType.DATE),
                    ],
                    vectorizer_config=Configure.Vectorizer.none()
                )
                logger.info("Created Document collection")
            else:
                logger.info("Document collection already exists")
            
            # Create Entity collection (for bridging with Neo4j)
            if "Entity" not in existing_collections:
                client.collections.create(
                    name="Entity",
                    description="Named entities extracted from documents",
                    properties=[
                        Property(name="entity_id", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="entity_type", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="entity_name", data_type=DataType.TEXT),
                        Property(name="canonical_name", data_type=DataType.TEXT, skip_vectorization=True),
                        Property(name="aliases", data_type=DataType.TEXT_ARRAY, skip_vectorization=True),
                        Property(name="created_at", data_type=DataType.DATE),
                    ],
                    vectorizer_config=Configure.Vectorizer.none()
                )
                logger.info("Created Entity collection")
            else:
                logger.info("Entity collection already exists")
            
            client.close()
            logger.info("Weaviate setup completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Weaviate setup failed: {e}")
            return False
    
    def setup_neo4j(self) -> bool:
        """Setup Neo4j knowledge graph - UNCHANGED"""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available")
            return False
        
        logger.info("Setting up Neo4j database...")
        
        try:
            driver = GraphDatabase.driver(
                self.neo4j_config['uri'],
                auth=(self.neo4j_config['user'], self.neo4j_config['password'])
            )
            
            with driver.session() as session:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE",
                    "CREATE CONSTRAINT indicator_name_unique IF NOT EXISTS FOR (i:Indicator) REQUIRE i.name IS UNIQUE",
                    "CREATE CONSTRAINT district_name_unique IF NOT EXISTS FOR (d:District) REQUIRE d.name IS UNIQUE",
                    "CREATE CONSTRAINT year_value_unique IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE",
                    "CREATE CONSTRAINT source_name_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.warning(f"Constraint creation failed: {e}")
                
                # Create indexes
                indexes = [
                    "CREATE INDEX fact_indicator IF NOT EXISTS FOR (f:Fact) ON (f.indicator)",
                    "CREATE INDEX fact_district IF NOT EXISTS FOR (f:Fact) ON (f.district)",
                    "CREATE INDEX fact_year IF NOT EXISTS FOR (f:Fact) ON (f.year)",
                    "CREATE INDEX fact_value IF NOT EXISTS FOR (f:Fact) ON (f.value)",
                    "CREATE INDEX fact_source IF NOT EXISTS FOR (f:Fact) ON (f.source)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.warning(f"Index creation failed: {e}")
            
            driver.close()
            logger.info("Neo4j setup completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Neo4j setup failed: {e}")
            return False
    
    def test_connections(self) -> Dict[str, bool]:
        """Test database connections"""
        results = {}
        
        # Test Weaviate
        try:
            client = weaviate.connect_to_local(
                host=self.weaviate_url.replace('http://', '').replace('https://', ''),
                port=8080,
                grpc_port=50051
            )
            client.is_ready()
            client.close()
            results['weaviate'] = True
            logger.info("Weaviate connection: OK")
        except Exception as e:
            results['weaviate'] = False
            logger.error(f"Weaviate connection failed: {e}")
        
        # Test Neo4j
        if NEO4J_AVAILABLE:
            try:
                driver = GraphDatabase.driver(
                    self.neo4j_config['uri'],
                    auth=(self.neo4j_config['user'], self.neo4j_config['password'])
                )
                with driver.session() as session:
                    session.run("RETURN 1")
                driver.close()
                results['neo4j'] = True
                logger.info("Neo4j connection: OK")
            except Exception as e:
                results['neo4j'] = False
                logger.error(f"Neo4j connection failed: {e}")
        else:
            results['neo4j'] = False
            logger.warning("Neo4j driver not available")
        
        return results
    
    def install_dependencies(self) -> bool:
        """Install required Python dependencies"""
        logger.info("Installing Python dependencies...")
        
        required_packages = [
            "weaviate-client==4.5.1",  # NEW
            "neo4j",
            "sentence-transformers",
            "fastapi",
            "uvicorn",
            "streamlit",
            "plotly",
            "pandas",
            "numpy",
            "requests",
            "beautifulsoup4",
            "PyMuPDF",
            "camelot-py[cv]",
            "pytesseract",
            "layoutparser",
            "fuzzywuzzy",
            "python-levenshtein"
        ]
        
        try:
            for package in required_packages:
                logger.info(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.warning(f"Failed to install {package}: {result.stderr}")
                else:
                    logger.info(f"Successfully installed {package}")
            
            logger.info("Dependencies installation completed")
            return True
        
        except Exception as e:
            logger.error(f"Dependencies installation failed: {e}")
            return False
    
    def setup_system_requirements(self) -> bool:
        """Setup system requirements (Tesseract, etc.)"""
        logger.info("Setting up system requirements...")
        
        # Check if Tesseract is available
        try:
            result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Tesseract OCR: Available")
            else:
                logger.warning("Tesseract OCR: Not available")
                logger.info("Please install Tesseract OCR for better text extraction")
        except FileNotFoundError:
            logger.warning("Tesseract OCR: Not installed")
            logger.info("Please install Tesseract OCR for better text extraction")
        
        # Check if Ghostscript is available (for Camelot)
        try:
            result = subprocess.run(["gs", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Ghostscript: Available")
            else:
                logger.warning("Ghostscript: Not available")
                logger.info("Please install Ghostscript for table extraction")
        except FileNotFoundError:
            logger.warning("Ghostscript: Not installed")
            logger.info("Please install Ghostscript for table extraction")
        
        return True
    
    def generate_setup_report(self) -> Dict[str, Any]:
        """Generate setup report"""
        report = {
            "setup_info": {
                "timestamp": datetime.now().isoformat(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "working_directory": str(Path.cwd())
            },
            "database_config": {
                "weaviate": {"url": self.weaviate_url},
                "neo4j": self.neo4j_config
            },
            "connection_test": self.test_connections()
        }
        
        return report

def main():
    """Main function to run database setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup databases for AP Education Policy Intelligence')
    parser.add_argument('--weaviate-only', action='store_true', help='Setup only Weaviate')
    parser.add_argument('--neo4j-only', action='store_true', help='Setup only Neo4j')
    parser.add_argument('--test-only', action='store_true', help='Only test connections')
    parser.add_argument('--install-deps', action='store_true', help='Install Python dependencies')
    parser.add_argument('--system-reqs', action='store_true', help='Check system requirements')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize setup
    setup = DatabaseSetup()
    
    try:
        if args.test_only:
            # Only test connections
            results = setup.test_connections()
            print("\nConnection Test Results:")
            for db, status in results.items():
                print(f"  {db}: {'✅ OK' if status else '❌ FAILED'}")
            sys.exit(0 if all(results.values()) else 1)
        
        elif args.install_deps:
            # Only install dependencies
            success = setup.install_dependencies()
            sys.exit(0 if success else 1)
        
        elif args.system_reqs:
            # Only check system requirements
            setup.setup_system_requirements()
            sys.exit(0)
        
        else:
            # Full setup
            success = True
            
            if not args.neo4j_only:
                success &= setup.setup_weaviate()
            
            if not args.weaviate_only:
                success &= setup.setup_neo4j()
            
            if success:
                # Test connections
                results = setup.test_connections()
                
                # Generate report
                report = setup.generate_setup_report()
                
                print("\n" + "="*60)
                print("DATABASE SETUP COMPLETED")
                print("="*60)
                print(f"Weaviate: {'✅ OK' if results.get('weaviate', False) else '❌ FAILED'}")
                print(f"Neo4j: {'✅ OK' if results.get('neo4j', False) else '❌ FAILED'}")
                print("="*60)
                
                if all(results.values()):
                    print("\n✅ All databases are ready for the pipeline!")
                else:
                    print("\n❌ Some databases failed. Please check the logs.")
                
                sys.exit(0 if all(results.values()) else 1)
            else:
                logger.error("Database setup failed")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Setup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()