#!/usr/bin/env python3
"""
Database Setup Utility
Sets up PostgreSQL and Neo4j databases for the pipeline
"""
import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Neo4j
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Database setup utility for AP education policy intelligence"""
    
    def __init__(self):
        # Database configurations
        self.pg_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'postgres',
            'password': 'password'
        }
        
        self.neo4j_config = {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password'
        }
        
        # Database names
        self.pg_database = 'ap_education_policy'
        self.neo4j_database = 'neo4j'
    
    def setup_postgresql(self) -> bool:
        """Setup PostgreSQL database with pgvector extension"""
        logger.info("Setting up PostgreSQL database...")
        
        try:
            # Connect to PostgreSQL server
            conn = psycopg2.connect(**self.pg_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Create database
            cursor.execute(f"CREATE DATABASE {self.pg_database};")
            logger.info(f"Created database: {self.pg_database}")
            
            cursor.close()
            conn.close()
            
            # Connect to the new database
            db_config = self.pg_config.copy()
            db_config['database'] = self.pg_database
            
            conn = psycopg2.connect(**db_config)
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("Enabled pgvector extension")
            
            # Create tables
            self._create_postgresql_tables(cursor)
            
            cursor.close()
            conn.close()
            
            logger.info("PostgreSQL setup completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"PostgreSQL setup failed: {e}")
            return False
    
    def _create_postgresql_tables(self, cursor):
        """Create PostgreSQL tables"""
        tables = {
            'facts': """
                CREATE TABLE IF NOT EXISTS facts (
                    fact_id VARCHAR(50) PRIMARY KEY,
                    indicator VARCHAR(100) NOT NULL,
                    category VARCHAR(50),
                    district VARCHAR(100),
                    year VARCHAR(20),
                    value DECIMAL(15,6),
                    unit VARCHAR(20),
                    source VARCHAR(50),
                    page_ref INTEGER,
                    confidence DECIMAL(3,2),
                    table_id VARCHAR(50),
                    pdf_name VARCHAR(200),
                    span_text TEXT,
                    embedding VECTOR(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """,
            
            'documents': """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id VARCHAR(50) PRIMARY KEY,
                    filename VARCHAR(200) NOT NULL,
                    source_type VARCHAR(50),
                    year VARCHAR(20),
                    total_pages INTEGER,
                    extraction_method VARCHAR(50),
                    checksum VARCHAR(64),
                    file_path VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """,
            
            'entities': """
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id VARCHAR(50) PRIMARY KEY,
                    entity_type VARCHAR(50) NOT NULL,
                    entity_name VARCHAR(200) NOT NULL,
                    canonical_name VARCHAR(200),
                    aliases TEXT[],
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
        }
        
        for table_name, table_sql in tables.items():
            try:
                cursor.execute(table_sql)
                logger.info(f"Created table: {table_name}")
            except Exception as e:
                logger.warning(f"Failed to create table {table_name}: {e}")
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_facts_indicator ON facts(indicator);",
            "CREATE INDEX IF NOT EXISTS idx_facts_district ON facts(district);",
            "CREATE INDEX IF NOT EXISTS idx_facts_year ON facts(year);",
            "CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source);",
            "CREATE INDEX IF NOT EXISTS idx_facts_embedding ON facts USING ivfflat (embedding vector_cosine_ops);",
            "CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_type);",
            "CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year);",
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);",
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(entity_name);"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
    
    def setup_neo4j(self) -> bool:
        """Setup Neo4j database"""
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available")
            return False
        
        logger.info("Setting up Neo4j database...")
        
        try:
            # Connect to Neo4j
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
        
        # Test PostgreSQL
        try:
            db_config = self.pg_config.copy()
            db_config['database'] = self.pg_database
            conn = psycopg2.connect(**db_config)
            conn.close()
            results['postgresql'] = True
            logger.info("PostgreSQL connection: OK")
        except Exception as e:
            results['postgresql'] = False
            logger.error(f"PostgreSQL connection failed: {e}")
        
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
            "psycopg2-binary",
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
                "postgresql": self.pg_config,
                "neo4j": self.neo4j_config
            },
            "connection_test": self.test_connections(),
            "system_requirements": {
                "tesseract": self._check_tesseract(),
                "ghostscript": self._check_ghostscript()
            }
        }
        
        return report
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available"""
        try:
            result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_ghostscript(self) -> bool:
        """Check if Ghostscript is available"""
        try:
            result = subprocess.run(["gs", "--version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

def main():
    """Main function to run database setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup databases for AP Education Policy Intelligence')
    parser.add_argument('--postgresql-only', action='store_true', help='Setup only PostgreSQL')
    parser.add_argument('--neo4j-only', action='store_true', help='Setup only Neo4j')
    parser.add_argument('--test-only', action='store_true', help='Only test connections')
    parser.add_argument('--install-deps', action='store_true', help='Install Python dependencies')
    parser.add_argument('--system-reqs', action='store_true', help='Check system requirements')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize setup
    setup = DatabaseSetup()
    
    try:
        if args.test_only:
            # Only test connections
            results = setup.test_connections()
            print("Connection Test Results:")
            for db, status in results.items():
                print(f"  {db}: {'OK' if status else 'FAILED'}")
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
                success &= setup.setup_postgresql()
            
            if not args.postgresql_only:
                success &= setup.setup_neo4j()
            
            if success:
                # Test connections
                results = setup.test_connections()
                
                # Generate report
                report = setup.generate_setup_report()
                
                print("\n" + "="*60)
                print("DATABASE SETUP COMPLETED")
                print("="*60)
                print(f"PostgreSQL: {'OK' if results.get('postgresql', False) else 'FAILED'}")
                print(f"Neo4j: {'OK' if results.get('neo4j', False) else 'FAILED'}")
                print("="*60)
                
                if all(results.values()):
                    print("All databases are ready for the pipeline!")
                else:
                    print("Some databases failed. Please check the logs.")
                
                sys.exit(0 if all(results.values()) else 1)
            else:
                logger.error("Database setup failed")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Setup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
