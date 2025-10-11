# AP Education Policy Intelligence Pipeline

Production-ready pipeline for Andhra Pradesh education policy intelligence, combining **data richness, graph reasoning, and LLM retrieval** for comprehensive policy analysis.

## 🎯 Overview

This pipeline transforms raw PDF documents into an **AI-ready data fabric** with:
- **Fact Graph**: Traceable facts connected to sources, years, districts, and policies
- **Vector Search**: Semantic similarity search using embeddings
- **Hybrid Retrieval**: Combining graph traversal and vector search
- **Real-time Analytics**: Interactive dashboard with trend analysis

## 🏗️ Architecture

```
Raw PDFs ─▶ Extraction Engine ─▶ Data Normalizer ─▶ Fact Graph (Neo4j)
                                       │
                                       ├─▶ Bridge Table (PostgreSQL + pgvector)
                                       │
                                       └─▶ RAG + Analytics Layer (LLM + Streamlit)
```

## 📁 Project Structure

```
pipeline/
├── stages/                          # Pipeline stages
│   ├── 1_extract_tables.py         # Table extraction from PDFs
│   ├── 2_normalize_schema.py       # Schema normalization
│   ├── 3_build_fact_table.py       # PostgreSQL bridge table
│   ├── 4_load_neo4j.py            # Neo4j knowledge graph
│   ├── 5_index_pgvector.py        # Vector embeddings
│   ├── 6_rag_api.py               # FastAPI RAG server
│   └── 7_dashboard_app.py         # Streamlit dashboard
├── utils/                          # Utility scripts
│   └── setup_database.py          # Database setup
├── validation/                     # Validation tools
│   └── validate_pipeline.py       # Pipeline validation
├── run_pipeline.py                 # Main orchestrator
└── README.md                       # This file

data/
├── raw/                           # Raw scraped data
├── extracted/                     # Extracted tables/text
├── normalized/                    # Normalized facts
├── bridge_table/                  # PostgreSQL exports
├── neo4j/                         # Neo4j exports
└── embeddings/                    # Vector embeddings
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.8+
python --version

# Install system dependencies
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr ghostscript

# macOS:
brew install tesseract ghostscript

# Windows:
# Download and install Tesseract OCR and Ghostscript
```

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Setup databases
python pipeline/utils/setup_database.py
```

### 3. Run Pipeline

```bash
# Full pipeline
python pipeline/run_pipeline.py

# Specific stages
python pipeline/run_pipeline.py --start-stage extract_tables --end-stage normalize_schema

# Dry run
python pipeline/run_pipeline.py --dry-run
```

### 4. Start Services

```bash
# Start RAG API server
python pipeline/stages/6_rag_api.py

# Start dashboard (in another terminal)
streamlit run pipeline/stages/7_dashboard_app.py
```

## 📊 Pipeline Stages

### Stage 1: Table Extraction
- **Input**: PDF documents
- **Output**: Structured tables and text
- **Tools**: Camelot, PyMuPDF, OCRmyPDF, LayoutParser
- **Validation**: Confidence scores, table structure checks

### Stage 2: Schema Normalization
- **Input**: Extracted tables/text
- **Output**: Unified fact schema
- **Features**: Fuzzy matching, canonical indicators, data validation
- **Validation**: Coverage metrics, data quality checks

### Stage 3: Fact Table Building
- **Input**: Normalized facts
- **Output**: PostgreSQL bridge table with pgvector
- **Features**: Vector embeddings, hybrid search indexes
- **Validation**: Data completeness, embedding coverage

### Stage 4: Neo4j Graph Loading
- **Input**: Normalized facts
- **Output**: Knowledge graph with relationships
- **Features**: Entity relationships, graph traversal
- **Validation**: Graph completeness, relationship integrity

### Stage 5: Vector Indexing
- **Input**: Fact table
- **Output**: Vector embeddings and indexes
- **Features**: Semantic search, similarity matching
- **Validation**: Embedding quality, search performance

### Stage 6: RAG API Server
- **Input**: Bridge table + Neo4j
- **Output**: REST API for hybrid retrieval
- **Features**: Semantic search, graph queries, trend analysis
- **Validation**: API performance, query latency

### Stage 7: Dashboard
- **Input**: RAG API
- **Output**: Interactive web dashboard
- **Features**: Search, trends, comparisons, analytics
- **Validation**: UI responsiveness, data visualization

## 🔍 Key Features

### Data Extraction
- **Multi-format Support**: PDFs, images, tables
- **OCR Integration**: Tesseract + LayoutParser
- **Table Detection**: Camelot for structured data
- **Text Extraction**: PyMuPDF for policy text

### Schema Normalization
- **Fuzzy Matching**: Indicator and district normalization
- **Data Validation**: Range checks, completeness validation
- **Canonical Forms**: Standardized terminology
- **Quality Metrics**: Confidence scoring

### Knowledge Graph
- **Entity Relationships**: Indicators, districts, years, sources
- **Graph Traversal**: Complex relationship queries
- **Ontology Design**: Structured knowledge representation
- **Inference**: Implicit relationship discovery

### Vector Search
- **Semantic Embeddings**: Sentence Transformers
- **Hybrid Retrieval**: Vector + keyword search
- **Similarity Matching**: Cosine similarity
- **Performance Optimization**: IVFFlat indexes

### RAG System
- **Hybrid Search**: Graph + vector integration
- **Query Processing**: Natural language understanding
- **Response Generation**: Contextual answers
- **Citation Tracking**: Source attribution

### Analytics Dashboard
- **Interactive Search**: Real-time querying
- **Trend Analysis**: Time-series visualization
- **District Comparison**: Comparative analysis
- **System Monitoring**: Health checks and statistics

## 📈 Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Extraction Accuracy | ≥95% | Correct table parsing |
| Indicator Coverage | ≥50 | Unique indicators |
| Provenance Completeness | 100% | Facts linked to sources |
| Query Latency | <3s | Retrieval + LLM response |
| Embedding Coverage | ≥90% | Facts with embeddings |
| Graph Completeness | ≥90% | Relationships loaded |

## 🛠️ Configuration

### Database Configuration
```python
# PostgreSQL
pg_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'ap_education_policy',
    'user': 'postgres',
    'password': 'password'
}

# Neo4j
neo4j_config = {
    'uri': 'bolt://localhost:7687',
    'user': 'neo4j',
    'password': 'password'
}
```

### Pipeline Configuration
```python
config = {
    "max_documents_per_source": 50,
    "extraction_confidence_threshold": 0.7,
    "normalization_similarity_threshold": 0.8,
    "embedding_model": "all-MiniLM-L6-v2",
    "vector_dimension": 384,
    "neo4j_batch_size": 1000,
    "api_port": 8000,
    "dashboard_port": 8501
}
```

## 🔧 Usage Examples

### Search Query
```python
# Semantic search
query = "GER in Visakhapatnam district"
results = rag_api.search(query, limit=10)

# Hybrid search with filters
filters = {"indicator": "GER", "year": "2020"}
results = rag_api.hybrid_search(query, filters=filters)
```

### Trend Analysis
```python
# Get trends for an indicator
trends = rag_api.trend_analysis("GER")

# Compare districts
comparison = rag_api.comparison_analysis("GER", "2020")
```

### Graph Queries
```cypher
// Find related indicators
MATCH (i:Indicator)-[:RELATED_TO]->(related:Indicator)
WHERE i.name = 'GER'
RETURN related.name

// District performance ranking
MATCH (f:Fact)-[:MEASURED_BY]->(i:Indicator {name: 'GER'})
MATCH (f)-[:LOCATED_IN]->(d:District)
MATCH (f)-[:OBSERVED_IN]->(y:Year {value: '2020'})
RETURN d.name, f.value
ORDER BY f.value DESC
```

## 🧪 Validation

### Pipeline Validation
```bash
# Full validation
python pipeline/validation/validate_pipeline.py

# Specific stage validation
python pipeline/validation/validate_pipeline.py --stage extraction
```

### Data Quality Checks
- **Extraction Accuracy**: Confidence scores
- **Normalization Coverage**: Unknown indicators
- **Fact Completeness**: Missing values
- **Embedding Coverage**: Vector availability
- **Graph Completeness**: Relationship integrity
- **Query Latency**: API performance

## 🚨 Troubleshooting

### Common Issues

1. **Extraction Failures**
   - Check PDF quality and format
   - Verify Tesseract installation
   - Adjust confidence thresholds

2. **Normalization Issues**
   - Review fuzzy matching thresholds
   - Check indicator lookup tables
   - Validate data ranges

3. **Database Connection Errors**
   - Verify PostgreSQL/Neo4j services
   - Check connection credentials
   - Test network connectivity

4. **API Performance Issues**
   - Optimize vector indexes
   - Check query complexity
   - Monitor resource usage

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python pipeline/run_pipeline.py

# Dry run for testing
python pipeline/run_pipeline.py --dry-run
```

## 📚 API Documentation

### Search Endpoint
```http
POST /search
Content-Type: application/json

{
    "query": "GER in Visakhapatnam",
    "limit": 10,
    "include_graph": true,
    "include_vector": true,
    "filters": {
        "indicator": "GER",
        "district": "Visakhapatnam",
        "year": "2020"
    }
}
```

### Statistics Endpoint
```http
GET /statistics
```

### Health Check
```http
GET /health
```

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: Telugu, Hindi
- **Real-time Updates**: Live data ingestion
- **Advanced Analytics**: ML-based insights
- **Mobile App**: Cross-platform access
- **API Rate Limiting**: Production scaling
- **Caching Layer**: Redis integration

### Integration Opportunities
- **UDISE+ Data**: National education database
- **NEP 2020**: Policy implementation tracking
- **Budget Data**: Financial analysis
- **Legal Documents**: Court judgments
- **Social Media**: Public sentiment

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and validation
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the validation reports
- Contact the development team

---

**AP Education Policy Intelligence Pipeline** - Transforming education data into actionable insights through AI-powered analysis.
