# AI Policy Co-Pilot MVP - Andhra Pradesh School Education

A comprehensive AI-powered policy intelligence system for Andhra Pradesh school education, built with RAG (Retrieval-Augmented Generation) and Knowledge Graph technologies.

## 🎯 Overview

This system provides intelligent querying and analysis of education policies, government orders, circulars, and related documents for Andhra Pradesh. It combines:

- **Vector Search**: Semantic similarity search using embeddings
- **Knowledge Graph**: Entity relationships and context understanding
- **NLP Pipeline**: Named Entity Recognition, Relation Extraction, Entity Linking, and Entity Resolution
- **Multi-source Data**: GOs, CSE portal, SCERT materials, Acts, Judgments, and Datasets

## 🏗️ Architecture

**Streamlined Production Architecture:**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  NLP Pipeline   │    │  Vector Store   │
│                 │    │                 │    │                 │
│ • GOs           │───▶│ • NER           │───▶│ • PostgreSQL    │
│ • CSE Portal    │    │ • RE            │    │ • pgvector      │
│ • SCERT         │    │ • EL            │    │ • Embeddings    │
│ • Acts          │    │ • ER            │    │                 │
│ • Judgments     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Scrapers      │    │  Bridge Table   │    │ Knowledge Graph │
│                 │    │                 │    │                 │
│ • GO Scraper    │    │ • Entity Links  │    │ • Neo4j         │
│ • CSE Scraper   │    │ • Span Mapping  │    │ • Ontology      │
│ • SCERT Scraper │    │ • Confidence    │    │ • Relationships │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Streamlit     │    │   Docker        │
│   Backend       │    │   Frontend      │    │   Deployment    │
│                 │    │                 │    │                 │
│ • REST API      │    │ • Query UI      │    │ • Compose       │
│ • RAG System    │    │ • Analytics     │    │ • Services      │
│ • Graph Queries │    │ • Visualization │    │ • Volumes       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 📁 Project Structure

```
policy-copilot/
├── backend/                          # Production API layer
│   ├── main.py                       # FastAPI server
│   ├── retriever.py                  # Vector search (PostgreSQL)
│   ├── embeddings.py                 # Embedding service
│   ├── graph_manager.py              # Neo4j operations
│   └── bridge_table.py               # PostgreSQL bridge table
├── data_pipeline/                    # Data processing layer
│   ├── scrapers/                     # Web scrapers
│   │   ├── go_scraper.py
│   │   ├── cse_scraper.py
│   │   └── scert_scraper.py
│   └── processors/
│       ├── text_extractor.py         # Advanced PDF/OCR extraction
│       └── nlp_processor.py          # NER, RE with InLegalBERT
├── graph_db/                         # Graph database layer
│   ├── ontology_schema.cql           # Neo4j schema
│   └── neo4j_loader.py               # Bulk loading operations
├── vector_db/                        # Vector database layer
│   └── init_pgvector.sql             # PostgreSQL setup
├── ui/                               # User interface
│   └── app.py                        # Streamlit UI
├── policy_intelligence/               # Pipeline orchestration
│   ├── main_pipeline.py              # Main pipeline orchestrator
│   ├── cli.py                        # Command-line interface
│   ├── config/settings.py            # Configuration
│   └── data/                         # Sample data
└── docker-compose.yml               # Docker services
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- PostgreSQL 15+ with pgvector extension
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

4. **Initialize the databases**
   ```bash
   # PostgreSQL will be initialized automatically
   # Neo4j schema will be loaded automatically
   ```

5. **Access the application**
   - **Streamlit UI**: http://localhost:8501
   - **FastAPI Backend**: http://localhost:8000
   - **Neo4j Browser**: http://localhost:7474
   - **PostgreSQL**: localhost:5432

### Manual Setup (without Docker)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up PostgreSQL with pgvector**
   ```bash
   # Install PostgreSQL and pgvector extension
   psql -U postgres -d policy -f vector_db/init_pgvector.sql
   ```

3. **Set up Neo4j**
   ```bash
   # Install Neo4j and load schema
   neo4j start
   # Load schema: graph_db/ontology_schema.cql
   ```

4. **Run the services**
   ```bash
   # Backend API
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   
   # Streamlit UI
   streamlit run ui/app.py --server.port 8501
   ```

## 📊 Data Pipeline

### 1. Data Scraping

The system scrapes data from multiple sources:

- **Government Orders (GOs)**: `data_pipeline/scrapers/go_scraper.py`
- **CSE Portal**: `data_pipeline/scrapers/cse_scraper.py`
- **SCERT Materials**: `data_pipeline/scrapers/scert_scraper.py`

```bash
# Run scrapers
python data_pipeline/scrapers/go_scraper.py
python data_pipeline/scrapers/cse_scraper.py
python data_pipeline/scrapers/scert_scraper.py
```

### 2. Text Extraction

Extract text from PDFs and other documents:

```bash
python data_pipeline/processors/text_extractor.py
```

### 3. NLP Processing

Process documents through the NLP pipeline:

```bash
python data_pipeline/processors/nlp_processor.py
```

### 4. Vector Embeddings

Generate and store embeddings:

```bash
python vector_db/embedding_loader.py
```

### 5. Knowledge Graph

Load entities and relations into Neo4j:

```bash
python graph_db/neo4j_loader.py
```

## 🔍 Usage

### Query Interface

1. **Open Streamlit UI**: http://localhost:8501
2. **Enter your query**: "What are the guidelines for teacher recruitment?"
3. **View results**: Relevant documents with confidence scores
4. **Explore graph context**: Related entities and relationships

### API Usage

```python
import requests

# Query the API
response = requests.post("http://localhost:8000/query", json={
    "query": "teacher recruitment guidelines",
    "max_results": 5,
    "include_graph": True
})

results = response.json()
print(results)
```

### Graph Queries

```cypher
// Find all policies related to teacher recruitment
MATCH (p:Policy)-[r]->(e:Entity)
WHERE e.name CONTAINS "teacher" AND e.name CONTAINS "recruitment"
RETURN p, r, e

// Get entity context
MATCH (e:Entity {id: "teacher_recruitment"})-[r*1..2]-(connected)
RETURN e, r, connected
```

## 📈 Analytics

The system provides comprehensive analytics:

- **Document Statistics**: Count by type, source, processing date
- **Entity Distribution**: Entity types and confidence scores
- **Relation Analysis**: Relationship patterns and frequencies
- **Graph Metrics**: Node counts, relationship density
- **Query Performance**: Response times and result quality

## 🛠️ Configuration

### Environment Variables

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=policy
POSTGRES_USER=postgres
POSTGRES_PASSWORD=1234

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
SIMILARITY_THRESHOLD=0.7
MAX_RESULTS=10
```

### Model Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **NER Model**: `law-ai/InLegalBERT` (legal domain)
- **Relation Extraction**: Pattern-based + ML models
- **Language Support**: English and Telugu

## 🧪 Testing

### Unit Tests

```bash
python -m pytest tests/
```

### Integration Tests

```bash
python test_pipeline.py
```

### Performance Tests

```bash
python tests/performance_test.py
```

## 📚 API Documentation

### FastAPI Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /query` - Query the policy database
- `GET /stats` - System statistics
- `GET /documents` - List documents
- `GET /documents/{doc_id}` - Get document details

### Query Parameters

```json
{
  "query": "string",
  "max_results": 5,
  "include_graph": true,
  "include_vector": true
}
```

### Response Format

```json
{
  "query": "string",
  "results": [
    {
      "doc_id": "string",
      "span_text": "string",
      "entity_id": "string",
      "confidence": 0.95,
      "source_url": "string"
    }
  ],
  "graph_context": [
    {
      "entity": {...},
      "connected_nodes": [...],
      "relationships": [...]
    }
  ],
  "processing_time": 1.23
}
```

## 🔧 Development

## 🔧 Recent Updates

### Codebase Cleanup (Latest)
- **Removed redundant implementations**: Eliminated duplicate code in `policy_intelligence/src/`
- **Consolidated functionality**: Single source of truth for each component
- **Production-ready**: PostgreSQL + Neo4j stack for scalability
- **Optional dependencies**: Graceful handling of missing packages (easyocr, transformers, neo4j)
- **Streamlined architecture**: Clear separation between data processing and API layers

### Key Changes
- ✅ Removed 6 redundant files from `policy_intelligence/src/`
- ✅ Updated imports to use production components
- ✅ Made optional dependencies (easyocr, transformers, neo4j) gracefully handled
- ✅ Maintained backward compatibility with existing pipeline
- ✅ Reduced codebase by ~2,000+ lines of duplicate code

### Adding New Data Sources

1. **Create scraper**: `data_pipeline/scrapers/new_source_scraper.py`
2. **Add text extraction**: Update `text_extractor.py`
3. **Configure NLP**: Add patterns to `nlp_processor.py`
4. **Update schema**: Modify `ontology_schema.cql`

### Adding New Entity Types

1. **Update NLP patterns**: `data_pipeline/processors/nlp_processor.py`
2. **Add to ontology**: `graph_db/ontology_schema.cql`
3. **Update mapping**: `backend/graph_manager.py`

## 🚀 Deployment

### Production Deployment

1. **Use production Docker images**
2. **Configure environment variables**
3. **Set up SSL certificates**
4. **Configure reverse proxy**
5. **Set up monitoring**

### Cloud Deployment

- **AWS**: Use ECS/EKS with RDS and Neptune
- **Azure**: Use Container Instances with Azure Database
- **GCP**: Use Cloud Run with Cloud SQL and Neo4j Aura

### Scaling

- **Horizontal scaling**: Multiple API instances
- **Database scaling**: Read replicas, connection pooling
- **Caching**: Redis for query results
- **CDN**: For static assets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Andhra Pradesh Government** for policy documents
- **Hugging Face** for transformer models
- **Neo4j** for graph database
- **PostgreSQL** for vector database
- **Streamlit** for UI framework

## 📞 Support

For support and questions:

- **Issues**: GitHub Issues
- **Documentation**: Wiki
- **Email**: support@policy-copilot.com

## 🔮 Roadmap

### Phase 1 (Current)
- ✅ Basic RAG system
- ✅ Knowledge graph
- ✅ Multi-source scraping
- ✅ Streamlit UI

### Phase 2 (Next)
- 🔄 Multilingual support (Telugu)
- 🔄 Advanced NLP models
- 🔄 Conflict detection
- 🔄 Policy impact analysis

### Phase 3 (Future)
- 📋 Higher education policies
- 📋 Vocational education
- 📋 Real-time updates
- 📋 Mobile application

---

**Built with ❤️ for Andhra Pradesh Education**