# Weaviate Migration Guide: Complete Codebase Transformation

## Executive Summary

This document provides **every single code change** needed to migrate the AP Policy Co-Pilot from PostgreSQL+pgvector to **Weaviate** as the vector database. Weaviate offers:

- ✅ **Simpler setup** - Single Docker container vs PostgreSQL+pgvector extension
- ✅ **Better vector search** - Purpose-built for semantic search
- ✅ **Built-in hybrid search** - BM25 + vector without custom code
- ✅ **GraphQL API** - Modern query interface
- ✅ **Better scalability** - Cloud-native architecture
- ✅ **Multi-tenancy** - Better for production deployments

---

## Architecture Change Overview

### **Before (Current):**
```
PostgreSQL + pgvector (vector storage)
    +
Neo4j (knowledge graph)
    +
Custom RAG layer
```

### **After (Weaviate):**
```
Weaviate (vector storage + hybrid search + filtering)
    +
Neo4j (knowledge graph) [KEEP - for entity relationships]
    +
Simplified RAG layer
```

**Key Decision:** We'll **KEEP Neo4j** for the knowledge graph (entity relationships, policy connections) and use Weaviate **ONLY** for vector search and fact retrieval.

---

## Part 1: Infrastructure Changes

### **1.1 Docker Compose - Complete Replacement**

**File:** `docker-compose.yml`

**ACTION:** Replace the entire PostgreSQL service with Weaviate

```yaml
version: '3.8'

services:
  # REMOVE THIS ENTIRE BLOCK:
  # postgres:
  #   image: ankane/pgvector:latest
  #   ...

  # ADD THIS NEW SERVICE:
  weaviate:
    image: semitechnologies/weaviate:1.24.1
    container_name: ap_policy_weaviate
    restart: unless-stopped
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'  # We'll provide our own embeddings
      ENABLE_MODULES: 'text2vec-transformers,generative-openai'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate
    networks:
      - policy_network

  # KEEP Neo4j AS-IS:
  neo4j:
    image: neo4j:5.15
    container_name: ap_policy_neo4j
    restart: unless-stopped
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_memory_heap_max__size: 2G
      NEO4J_dbms_memory_pagecache_size: 1G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    networks:
      - policy_network

  # KEEP Backend, UI, etc. - just update their environment variables
  backend:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: ap_policy_backend
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      # REMOVE PostgreSQL vars:
      # POSTGRES_HOST: postgres
      # POSTGRES_PORT: 5432
      # ...
      
      # ADD Weaviate vars:
      WEAVIATE_URL: http://weaviate:8080
      WEAVIATE_TIMEOUT: 30
      
      # KEEP Neo4j vars:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: password
    depends_on:
      - weaviate  # Changed from postgres
      - neo4j
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    networks:
      - policy_network

  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    container_name: ap_policy_ui
    restart: unless-stopped
    ports:
      - "8501:8501"
    environment:
      BACKEND_URL: http://backend:8000
    depends_on:
      - backend
    networks:
      - policy_network

volumes:
  weaviate_data:  # NEW
  neo4j_data:     # KEEP
  neo4j_logs:     # KEEP

networks:
  policy_network:
    driver: bridge
```

---

### **1.2 Environment Variables**

**File:** `.env`

**ACTION:** Replace PostgreSQL variables with Weaviate variables

```bash
# REMOVE THESE:
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DB=ap_education_policy
# POSTGRES_USER=postgres
# POSTGRES_PASSWORD=password

# ADD THESE:
WEAVIATE_URL=http://localhost:8080
WEAVIATE_TIMEOUT=30
WEAVIATE_BATCH_SIZE=100

# KEEP THESE:
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Model Configuration (no change)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
SIMILARITY_THRESHOLD=0.7
MAX_RESULTS=10
```

---

### **1.3 Requirements.txt**

**File:** `requirements.txt`

**ACTION:** Replace psycopg2 with weaviate-client

```txt
# REMOVE THIS:
# psycopg2-binary==2.9.9

# ADD THIS:
weaviate-client==4.5.1

# KEEP THESE:
neo4j==5.15.0
sentence-transformers==2.2.2
fastapi==0.109.0
uvicorn==0.27.0
streamlit==1.31.0
plotly==5.18.0
pandas==2.2.0
numpy==1.26.3
requests==2.31.0
beautifulsoup4==4.12.3
PyMuPDF==1.23.8
camelot-py[cv]==0.11.0
pytesseract==0.3.10
fuzzywuzzy==0.18.0
python-levenshtein==0.25.0
pydantic==2.5.3
python-dotenv==1.0.0
```

---

## Part 2: Database Setup Changes

### **2.1 Setup Database Script - Complete Rewrite**

**File:** `pipeline/utils/setup_database.py`

**ACTION:** Replace entire file

```python
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
```

---

## Part 3: Pipeline Stage Changes

### **3.1 Stage 3: Fact Table Builder - Complete Rewrite**

**File:** `pipeline/stages/3_build_fact_table.py`

**ACTION:** Replace PostgreSQL code with Weaviate

```python
#!/usr/bin/env python3
"""
Stage 3: Fact Table Builder (Weaviate)
Loads normalized facts into Weaviate vector database
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import uuid

import weaviate
from weaviate.classes.data import DataObject
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class WeaviateFactLoader:
    """Load facts into Weaviate"""
    
    def __init__(self, output_dir: str = "data/weaviate"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Weaviate connection
        weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        self.client = weaviate.connect_to_local(
            host=weaviate_url.replace('http://', '').replace('https://', ''),
            port=8080,
            grpc_port=50051
        )
        
        # Embedding model
        model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        
        logger.info(f"Initialized Weaviate loader with {model_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not text or text.strip() == "":
            # Return zero vector for empty text
            return [0.0] * 384
        
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def load_facts(self, normalized_facts: List[Dict[str, Any]]) -> bool:
        """Load facts into Weaviate"""
        logger.info(f"Loading {len(normalized_facts)} facts into Weaviate...")
        
        try:
            fact_collection = self.client.collections.get("Fact")
            
            # Batch insert
            with fact_collection.batch.dynamic() as batch:
                for idx, fact in enumerate(normalized_facts):
                    if idx % 100 == 0:
                        logger.info(f"Processing fact {idx}/{len(normalized_facts)}")
                    
                    # Generate embedding from span_text
                    span_text = fact.get('span_text', '')
                    embedding = self.generate_embedding(span_text)
                    
                    # Create data object
                    data_object = {
                        "fact_id": fact.get('fact_id', str(uuid.uuid4())),
                        "indicator": fact.get('indicator', 'Unknown'),
                        "category": fact.get('category', 'Unknown'),
                        "district": fact.get('district', 'Unknown'),
                        "year": fact.get('year', 'Unknown'),
                        "value": float(fact.get('value', 0.0)) if fact.get('value') is not None else 0.0,
                        "unit": fact.get('unit', ''),
                        "source": fact.get('source', ''),
                        "page_ref": int(fact.get('page_ref', 0)) if fact.get('page_ref') else 0,
                        "confidence": float(fact.get('confidence', 0.0)),
                        "table_id": fact.get('table_id', ''),
                        "pdf_name": fact.get('pdf_name', ''),
                        "span_text": span_text,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Add to batch with vector
                    batch.add_object(
                        properties=data_object,
                        vector=embedding
                    )
            
            logger.info(f"✅ Successfully loaded {len(normalized_facts)} facts")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load facts: {e}")
            return False
    
    def load_documents(self, extracted_data: List[Dict[str, Any]]) -> bool:
        """Load document metadata into Weaviate"""
        logger.info(f"Loading {len(extracted_data)} documents into Weaviate...")
        
        try:
            doc_collection = self.client.collections.get("Document")
            
            # Extract unique documents
            documents = {}
            for item in extracted_data:
                doc_id = item.get('doc_id')
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "doc_id": doc_id,
                        "filename": item.get('filename', ''),
                        "source_type": item.get('source_type', ''),
                        "year": item.get('year', ''),
                        "total_pages": int(item.get('page', 1)),
                        "extraction_method": item.get('extraction_method', ''),
                        "checksum": item.get('checksum', ''),
                        "file_path": item.get('file_path', ''),
                        "created_at": datetime.now().isoformat()
                    }
            
            # Batch insert
            with doc_collection.batch.dynamic() as batch:
                for doc_data in documents.values():
                    batch.add_object(properties=doc_data)
            
            logger.info(f"✅ Successfully loaded {len(documents)} documents")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return False
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary of loaded data"""
        try:
            fact_collection = self.client.collections.get("Fact")
            doc_collection = self.client.collections.get("Document")
            
            fact_count = fact_collection.aggregate.over_all(total_count=True).total_count
            doc_count = doc_collection.aggregate.over_all(total_count=True).total_count
            
            summary = {
                "total_facts": fact_count,
                "total_documents": doc_count,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save summary
            summary_file = self.output_dir / "weaviate_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            return summary
        
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {}
    
    def close(self):
        """Close Weaviate connection"""
        self.client.close()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load facts into Weaviate')
    parser.add_argument('--normalized-file', default='data/normalized/normalized_facts.json')
    parser.add_argument('--extracted-file', default='data/extracted/all_extracted_data.json')
    parser.add_argument('--output-dir', default='data/weaviate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    try:
        with open(args.normalized_file, 'r') as f:
            normalized_facts = json.load(f)
        logger.info(f"Loaded {len(normalized_facts)} normalized facts")
    except FileNotFoundError:
        logger.error(f"Normalized facts file not found: {args.normalized_file}")
        return
    
    try:
        with open(args.extracted_file, 'r') as f:
            extracted_data = json.load(f)
        logger.info(f"Loaded {len(extracted_data)} extracted items")
    except FileNotFoundError:
        logger.error(f"Extracted data file not found: {args.extracted_file}")
        return
    
    # Initialize loader
    loader = WeaviateFactLoader(output_dir=args.output_dir)
    
    try:
        # Load facts and documents
        success = True
        success &= loader.load_facts(normalized_facts)
        success &= loader.load_documents(extracted_data)
        
        if success:
            # Generate summary
            summary = loader.generate_summary()
            
            print(f"\n✅ Weaviate Loading Complete:")
            print(f"   Total facts: {summary.get('total_facts', 0)}")
            print(f"   Total documents: {summary.get('total_documents', 0)}")
            print(f"   Output directory: {args.output_dir}")
        else:
            logger.error("Weaviate loading failed")
    
    finally:
        loader.close()

if __name__ == "__main__":
    main()
```

---

### **3.2 Stage 5: Vector Indexing - REMOVE/SIMPLIFY**

**File:** `pipeline/stages/5_index_pgvector.py`

**ACTION:** This stage is NO LONGER NEEDED with Weaviate (vectors are indexed automatically)

**Option 1: Delete the file completely**

**Option 2: Replace with a simple verification script:**

```python
#!/usr/bin/env python3
"""
Stage 5: Weaviate Index Verification
Verifies that Weaviate indexes are properly created
"""
import logging
import weaviate

logger = logging.getLogger(__name__)

def verify_weaviate_indexes():
    """Verify Weaviate collections and indexes"""
    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        
        collections = client.collections.list_all()
        
        print("\n✅ Weaviate Collections:")
        for collection in collections:
            print(f"   - {collection.name}")
            
            # Get collection stats
            col = client.collections.get(collection.name)
            count = col.aggregate.over_all(total_count=True).total_count
            print(f"     Objects: {count}")
        
        client.close()
        return True
    
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    verify_weaviate_indexes()
```

---

## Part 4: Backend API Changes

### **4.1 Retriever Module - Complete Rewrite**

**File:** `backend/retriever.py`

**ACTION:** Replace entire PostgreSQL-based retriever with Weaviate

```python
"""
Weaviate Retriever for AP Policy Co-Pilot
Hybrid search combining vector similarity and keyword filtering
"""
import os
import logging
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.query import MetadataQuery, Filter
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class WeaviateRetriever:
    """Hybrid retriever using Weaviate"""
    
    def __init__(self):
        # Connect to Weaviate
        weaviate_url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
        self.client = weaviate.connect_to_local(
            host=weaviate_url.replace('http://', '').replace('https://', ''),
            port=8080,
            grpc_port=50051
        )
        
        # Embedding model
        model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name)
        
        logger.info("Initialized Weaviate retriever")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query"""
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.7  # 0.7 = 70% vector, 30% keyword
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and keyword search
        
        Args:
            query: Search query
            limit: Maximum results
            filters: Property filters (indicator, district, year, etc.)
            alpha: Balance between vector (1.0) and keyword (0.0) search
        """
        try:
            fact_collection = self.client.collections.get("Fact")
            
            # Generate query embedding
            query_vector = self.generate_embedding(query)
            
            # Build filter if provided
            filter_obj = None
            if filters:
                filter_conditions = []
                
                if 'indicator' in filters:
                    filter_conditions.append(
                        Filter.by_property("indicator").equal(filters['indicator'])
                    )
                
                if 'district' in filters:
                    filter_conditions.append(
                        Filter.by_property("district").equal(filters['district'])
                    )
                
                if 'year' in filters:
                    filter_conditions.append(
                        Filter.by_property("year").equal(filters['year'])
                    )
                
                if 'category' in filters:
                    filter_conditions.append(
                        Filter.by_property("category").equal(filters['category'])
                    )
                
                # Combine filters with AND
                if filter_conditions:
                    filter_obj = filter_conditions[0]
                    for condition in filter_conditions[1:]:
                        filter_obj = filter_obj & condition
            
            # Hybrid search
            response = fact_collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=limit,
                filters=filter_obj,
                return_metadata=MetadataQuery(distance=True, score=True)
            )
            
            # Format results
            results = []
            for obj in response.objects:
                result = {
                    'fact_id': obj.properties.get('fact_id'),
                    'indicator': obj.properties.get('indicator'),
                    'category': obj.properties.get('category'),
                    'district': obj.properties.get('district'),
                    'year': obj.properties.get('year'),
                    'value': obj.properties.get('value'),
                    'unit': obj.properties.get('unit'),
                    'source': obj.properties.get('source'),
                    'page_ref': obj.properties.get('page_ref'),
                    'confidence': obj.properties.get('confidence'),
                    'span_text': obj.properties.get('span_text'),
                    'pdf_name': obj.properties.get('pdf_name'),
                    'score': obj.metadata.score if obj.metadata else None,
                    'distance': obj.metadata.distance if obj.metadata else None
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def vector_search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Pure vector similarity search"""
        return self.search(query, limit, filters, alpha=1.0)
    
    def keyword_search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Pure keyword (BM25) search"""
        return self.search(query, limit, filters, alpha=0.0)
    
    def get_by_id(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Get fact by ID"""
        try:
            fact_collection = self.client.collections.get("Fact")
            
            response = fact_collection.query.fetch_objects(
                filters=Filter.by_property("fact_id").equal(fact_id),
                limit=1
            )
            
            if response.objects:
                obj = response.objects[0]
                return {
                    'fact_id': obj.properties.get('fact_id'),
                    'indicator': obj.properties.get('indicator'),
                    'district': obj.properties.get('district'),
                    'year': obj.properties.get('year'),
                    'value': obj.properties.get('value'),
                    'unit': obj.properties.get('unit'),
                    'source': obj.properties.get('source'),
                    'span_text': obj.properties.get('span_text'),
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Get by ID failed: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            fact_collection = self.client.collections.get("Fact")
            doc_collection = self.client.collections.get("Document")
            
            fact_count = fact_collection.aggregate.over_all(total_count=True).total_count
            doc_count = doc_collection.aggregate.over_all(total_count=True).total_count
            
            # Get unique indicators
            indicator_response = fact_collection.aggregate.over_all(
                group_by="indicator"
            )
            unique_indicators = len(indicator_response.groups) if indicator_response.groups else 0
            
            # Get unique districts
            district_response = fact_collection.aggregate.over_all(
                group_by="district"
            )
            unique_districts = len(district_response.groups) if district_response.groups else 0
            
            return {
                "total_facts": fact_count,
                "total_documents": doc_count,
                "unique_indicators": unique_indicators,
                "unique_districts": unique_districts,
                "database": "Weaviate"
            }
        
        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            return {}
    
    def close(self):
        """Close connection"""
        self.client.close()
```

---

### **4.2 Main API - Update Imports**

**File:** `backend/main.py`

**ACTION:** Update imports and dependency injection

```python
"""
FastAPI Backend for AP Policy Co-Pilot
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time

# CHANGE THIS:
# from backend.retriever import VectorRetriever
# from backend.bridge_table import BridgeTableManager

# TO THIS:
from backend.retriever import WeaviateRetriever
from backend.graph_manager import GraphManager

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AP Education Policy Co-Pilot API",
    description="RAG-based policy intelligence system",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
retriever = None
graph_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global retriever, graph_manager
    
    logger.info("Initializing backend services...")
    
    # CHANGE THIS:
    # retriever = VectorRetriever()
    # TO THIS:
    retriever = WeaviateRetriever()
    
    graph_manager = GraphManager()
    
    logger.info("Backend services initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global retriever, graph_manager
    
    if retriever:
        retriever.close()
    if graph_manager:
        graph_manager.close()
    
    logger.info("Backend services closed")

# Request/Response models (unchanged)
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    include_graph: bool = True
    include_vector: bool = True
    filters: Optional[Dict[str, Any]] = None
    alpha: float = 0.7  # NEW: hybrid search balance

class SearchResult(BaseModel):
    fact_id: str
    indicator: str
    district: str
    year: str
    value: float
    unit: str
    source: str
    span_text: str
    score: Optional[float] = None
    distance: Optional[float] = None

class QueryResponse(BaseModel):
    query: str
    results: List[SearchResult]
    graph_context: Optional[List[Dict[str, Any]]] = None
    processing_time: float

# Endpoints (minimal changes)
@app.get("/")
async def root():
    return {
        "name": "AP Education Policy Co-Pilot API",
        "version": "2.0.0",
        "database": "Weaviate + Neo4j",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Weaviate
        stats = retriever.get_statistics()
        
        return {
            "status": "healthy",
            "database": "Weaviate",
            "facts_count": stats.get("total_facts", 0),
            "documents_count": stats.get("total_documents", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/search", response_model=QueryResponse)
async def search(request: SearchRequest):
    """Search endpoint with hybrid search"""
    try:
        start_time = time.time()
        
        # Weaviate hybrid search
        results = retriever.search(
            query=request.query,
            limit=request.limit,
            filters=request.filters,
            alpha=request.alpha
        )
        
        # Get graph context if requested
        graph_context = None
        if request.include_graph and results:
            entity_ids = [r['fact_id'] for r in results[:5]]
            graph_context = graph_manager.get_entity_context(entity_ids)
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            results=results,
            graph_context=graph_context,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """Get system statistics"""
    try:
        weaviate_stats = retriever.get_statistics()
        graph_stats = graph_manager.get_statistics()
        
        return {
            "weaviate": weaviate_stats,
            "knowledge_graph": graph_stats,
            "total_facts": weaviate_stats.get("total_facts", 0),
            "total_documents": weaviate_stats.get("total_documents", 0)
        }
    
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### **4.3 Bridge Table Module - REMOVE**

**File:** `backend/bridge_table.py`

**ACTION:** DELETE THIS FILE - No longer needed with Weaviate

The bridge table concept was specific to PostgreSQL. Weaviate handles this internally.

---

### **4.4 Embeddings Module - Simplify**

**File:** `backend/embeddings.py`

**ACTION:** Simplify (Weaviate handles most embedding logic)

```python
"""
Embedding Service
Generates embeddings for text using sentence-transformers
"""
import os
from typing import List
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Embedding generation service"""
    
    def __init__(self):
        model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.model = SentenceTransformer(model_name)
        self.dimension = int(os.getenv('EMBEDDING_DIMENSION', '384'))
        
        logger.info(f"Initialized embedding model: {model_name}")
    
    def encode(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        if not text or text.strip() == "":
            return [0.0] * self.dimension
        
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
```

---

## Part 5: Pipeline Orchestrator Changes

### **5.1 Update Pipeline Stages List**

**File:** `pipeline/run_pipeline.py`

**ACTION:** Update stage 3 and remove/simplify stage 5

```python
# Around line 20-50
self.stages = [
    {
        "name": "extract_tables",
        "script": "1_extract_tables.py",
        "description": "Extract tables and text from PDFs",
        "input": "data/preprocessed/documents",
        "output": "data/extracted"
    },
    {
        "name": "normalize_schema",
        "script": "2_normalize_schema.py",
        "description": "Normalize extracted data into unified schema",
        "input": "data/extracted/all_extracted_data.json",
        "output": "data/normalized"
    },
    {
        "name": "load_weaviate",  # CHANGED NAME
        "script": "3_build_fact_table.py",  # Same file, different implementation
        "description": "Load facts into Weaviate vector database",  # CHANGED
        "input": "data/normalized/normalized_facts.json",
        "output": "data/weaviate"  # CHANGED
    },
    {
        "name": "load_neo4j",
        "script": "4_load_neo4j.py",
        "description": "Load facts into Neo4j knowledge graph",
        "input": "data/normalized/normalized_facts.json",
        "output": "data/neo4j"
    },
    # REMOVE OR SIMPLIFY STAGE 5:
    # {
    #     "name": "index_pgvector",
    #     "script": "5_index_pgvector.py",
    #     ...
    # },
    {
        "name": "verify_indexes",  # OPTIONAL
        "script": "5_verify_weaviate.py",
        "description": "Verify Weaviate indexes",
        "input": "data/weaviate",
        "output": "data/weaviate"
    },
    {
        "name": "rag_api",
        "script": "6_rag_api.py",
        "description": "Start RAG API server",
        "input": "data/weaviate",  # CHANGED
        "output": "api_server"
    },
    {
        "name": "dashboard",
        "script": "7_dashboard_app.py",
        "description": "Launch interactive dashboard",
        "input": "api_server",
        "output": "dashboard"
    }
]
```

---

## Part 6: Documentation Updates

### **6.1 README.md Changes**

**File:** `README.md`

**ACTION:** Update architecture diagram and setup instructions

Find and replace:

```markdown
<!-- OLD -->
PostgreSQL + pgvector → Vector search

<!-- NEW -->
Weaviate → Vector search + hybrid retrieval
```

Update setup section:

```markdown
## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- **Weaviate** (via Docker)
- Neo4j 5.15+

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd policy-copilot
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - **Weaviate Console**: http://localhost:8080/v1/schema
   - **Neo4j Browser**: http://localhost:7474
   - **FastAPI Backend**: http://localhost:8000
   - **Streamlit UI**: http://localhost:8501
```

---

### **6.2 Pipeline README Updates**

**File:** `pipeline/README.md`

**ACTION:** Update architecture section

```markdown
## Architecture

```
Raw PDFs → Extraction → Normalization → Weaviate + Neo4j → RAG API → Dashboard
```

### Components

1. **Weaviate** - Vector database with hybrid search (replaces PostgreSQL+pgvector)
2. **Neo4j** - Knowledge graph for entity relationships
3. **FastAPI** - REST API for queries
4. **Streamlit** - Interactive dashboard
```

---

## Part 7: Testing & Validation

### **7.1 Create Weaviate Test Script**

**File:** `tests/test_weaviate.py`

**ACTION:** Create new test file

```python
"""
Test Weaviate integration
"""
import weaviate
from sentence_transformers import SentenceTransformer

def test_weaviate_connection():
    """Test connection to Weaviate"""
    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        assert client.is_ready()
        client.close()
        print("✅ Weaviate connection successful")
        return True
    except Exception as e:
        print(f"❌ Weaviate connection failed: {e}")
        return False

def test_schema_exists():
    """Test that required collections exist"""
    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        
        collections = [c.name for c in client.collections.list_all()]
        required = ["Fact", "Document", "Entity"]
        
        for col in required:
            assert col in collections, f"Missing collection: {col}"
            print(f"✅ Collection exists: {col}")
        
        client.close()
        return True
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False

def test_hybrid_search():
    """Test hybrid search functionality"""
    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate test embedding
        query = "education enrollment statistics"
        vector = model.encode(query).tolist()
        
        # Perform hybrid search
        fact_collection = client.collections.get("Fact")
        response = fact_collection.query.hybrid(
            query=query,
            vector=vector,
            alpha=0.7,
            limit=5
        )
        
        print(f"✅ Hybrid search returned {len(response.objects)} results")
        
        client.close()
        return True
    except Exception as e:
        print(f"❌ Hybrid search test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Weaviate Integration...\n")
    
    tests = [
        test_weaviate_connection,
        test_schema_exists,
        test_hybrid_search
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning: {test.__name__}")
        results.append(test())
    
    print("\n" + "="*50)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print("="*50)
```

---

## Part 8: Migration Checklist

### **Complete Migration Checklist**

```markdown
## Phase 1: Backup & Preparation
- [ ] Backup existing PostgreSQL data (if any)
- [ ] Document current data counts
- [ ] Create git branch: `git checkout -b weaviate-migration`

## Phase 2: Infrastructure
- [ ] Update `docker-compose.yml`
- [ ] Update `.env` file
- [ ] Update `requirements.txt`
- [ ] Run `docker-compose up -d` to start Weaviate
- [ ] Verify Weaviate is running: `curl http://localhost:8080/v1/.well-known/ready`

## Phase 3: Database Setup
- [ ] Replace `pipeline/utils/setup_database.py`
- [ ] Run `python pipeline/utils/setup_database.py`
- [ ] Test connections: `python pipeline/utils/setup_database.py --test-only`
- [ ] Verify schema in Weaviate console: http://localhost:8080/v1/schema

## Phase 4: Pipeline Stages
- [ ] Replace `pipeline/stages/3_build_fact_table.py`
- [ ] Remove/simplify `pipeline/stages/5_index_pgvector.py`
- [ ] Update `pipeline/run_pipeline.py` stage list
- [ ] Test Stage 3 in isolation

## Phase 5: Backend API
- [ ] Replace `backend/retriever.py`
- [ ] Update `backend/main.py` imports
- [ ] Delete `backend/bridge_table.py`
- [ ] Simplify `backend/embeddings.py`
- [ ] Test API: `curl http://localhost:8000/health`

## Phase 6: Testing
- [ ] Create `tests/test_weaviate.py`
- [ ] Run tests: `python tests/test_weaviate.py`
- [ ] Test end-to-end pipeline with sample data
- [ ] Verify search results quality

## Phase 7: Documentation
- [ ] Update `README.md`
- [ ] Update `pipeline/README.md`
- [ ] Create migration guide (this document)
- [ ] Update architecture diagrams

## Phase 8: Cleanup
- [ ] Remove all PostgreSQL references
- [ ] Remove unused imports
- [ ] Update deployment docs
- [ ] Commit changes: `git commit -m "Migrate to Weaviate"`
```

---

## Part 9: Quick Start Commands

### **Complete Migration in 10 Steps**

```bash
# 1. Create backup
git checkout -b weaviate-migration

# 2. Update dependencies
pip uninstall psycopg2-binary
pip install weaviate-client==4.5.1

# 3. Start Weaviate
docker-compose up -d weaviate

# 4. Verify Weaviate
curl http://localhost:8080/v1/.well-known/ready

# 5. Setup database
python pipeline/utils/setup_database.py

# 6. Test connection
python pipeline/utils/setup_database.py --test-only

# 7. Run pipeline (with existing normalized data)
python pipeline/stages/3_build_fact_table.py

# 8. Verify data loaded
python tests/test_weaviate.py

# 9. Start API
python backend/main.py

# 10. Test API
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "education statistics", "limit": 5}'
```

---

## Part 10: Troubleshooting

### **Common Issues & Solutions**

#### Issue 1: Weaviate not starting
```bash
# Check logs
docker logs ap_policy_weaviate

# Solution: Increase Docker memory
# Docker Desktop → Settings → Resources → Memory: 4GB minimum
```

#### Issue 2: Import errors
```bash
# Error: No module named 'weaviate'
pip install weaviate-client==4.5.1

# Error: No module named 'psycopg2'
# This is expected! Remove all psycopg2 imports
```

#### Issue 3: Connection refused
```bash
# Check if Weaviate is running
docker ps | grep weaviate

# Check port
netstat -an | grep 8080

# Restart Weaviate
docker-compose restart weaviate
```

#### Issue 4: Schema not created
```bash
# Manually verify schema
curl http://localhost:8080/v1/schema

# Recreate schema
python pipeline/utils/setup_database.py --weaviate-only
```

#### Issue 5: Search returns no results
```bash
# Check data count
curl http://localhost:8080/v1/objects/Fact

# Verify embeddings
# Check that vectors are being generated in Stage 3
```

---

## Part 11: Benefits Summary

### **Why Weaviate is Better for This Project**

1. **Simpler Setup**
   - No PostgreSQL extensions needed
   - No complex pgvector configuration
   - Single Docker container

2. **Better Search**
   - Built-in hybrid search (vector + BM25)
   - Better relevance ranking
   - Configurable alpha parameter

3. **Production Ready**
   - Cloud-native architecture
   - Better scaling (horizontal)
   - Built-in monitoring

4. **Developer Experience**
   - Modern Python client
   - GraphQL API
   - Better documentation

5. **Performance**
   - Faster vector indexing (HNSW)
   - Better query performance
   - Efficient batching

---

## Files Summary

### **Files to CREATE:**
- `tests/test_weaviate.py`

### **Files to REPLACE COMPLETELY:**
- `docker-compose.yml`
- `.env`
- `requirements.txt`
- `pipeline/utils/setup_database.py`
- `pipeline/stages/3_build_fact_table.py`
- `backend/retriever.py`

### **Files to UPDATE:**
- `backend/main.py` (imports only)
- `backend/embeddings.py` (simplify)
- `pipeline/run_pipeline.py` (stage list)
- `README.md` (architecture section)
- `pipeline/README.md` (setup instructions)

### **Files to DELETE:**
- `backend/bridge_table.py`
- `vector_db/init_pgvector.sql`
- `pipeline/stages/5_index_pgvector.py` (optional: replace with verification)

### **Files UNCHANGED:**
- All extraction and normalization stages (1, 2)
- Neo4j loader (stage 4)
- Graph manager
- All scrapers
- NLP processor
- Dashboard UI (minor API call changes)

---

## Estimated Migration Time

- **Small project (POC)**: 2-3 hours
- **Medium project**: 4-6 hours  
- **Production system**: 1-2 days (includes testing)

---

## Next Steps

1. Review this guide completely
2. Create backup branch
3. Start with Phase 1 (Infrastructure)
4. Test each phase before proceeding
5. Report any issues

Would you like me to:
1. Create a migration script that automates these changes?
2. Generate the actual replacement files?
3. Create a rollback plan?
