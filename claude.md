# AP Policy Co-Pilot: Data Processing Pipeline with Weaviate

## Executive Summary

This system is an **AI-powered policy intelligence platform** for Andhra Pradesh education data. The architecture has been **migrated from PostgreSQL+pgvector to Weaviate** for better vector search capabilities and simpler deployment.

### **Current Architecture**
```
Raw PDFs → Extraction → Normalization → Weaviate + Neo4j → RAG API → Dashboard
```

**Key Components:**
- 🚀 **Weaviate** - Vector database with hybrid search (vector + BM25)
- 🕸️ **Neo4j** - Knowledge graph for entity relationships  
- 🔍 **RAG API** - FastAPI-based search and retrieval
- 📊 **Dashboard** - Streamlit interactive interface

---

## 🚨 Critical Issues Identified

### **1. Missing Raw Data (BLOCKER)**

**Problem:** The pipeline expects PDFs in `data/preprocessed/documents/`, but this directory is likely empty or doesn't exist.

**Evidence:**
- `.gitignore` excludes all PDF files: `*.pdf`
- README instructs to scrape data, but scrapers may not have been run
- Pipeline stage 1 (`1_extract_tables.py`) looks for PDFs in `data/preprocessed/documents`

**Impact:** Pipeline cannot start extraction without source documents.

**Solution:**
```bash
# Check if data exists
ls -la data/preprocessed/documents/

# If empty, you need to either:
# 1. Manually download PDFs from government websites
# 2. Run the scrapers (but they may also be broken)
# 3. Copy sample data into this directory
```

---

### **2. Database Not Initialized (BLOCKER)**

**Problem:** Weaviate and Neo4j databases are not set up or running.

**Evidence:**
- Stage 3 (`3_build_fact_table.py`) tries to connect to Weaviate
- Stage 4 (`4_load_neo4j.py`) tries to connect to Neo4j
- Default connection strings expect databases at `localhost:8080` and `localhost:7687`

**Expected Configuration:**
```python
# Weaviate
{
    'url': 'http://localhost:8080',
    'timeout': 30
}

# Neo4j
{
    'uri': 'bolt://localhost:7687',
    'user': 'neo4j',
    'password': 'password'
}
```

**Impact:** Stages 3+ will fail immediately with connection errors.

**Solution:**
```bash
# Start services with Docker Compose
docker-compose up -d

# Run the database setup script  
python pipeline/utils/setup_database.py

# Or manually:
# 1. Start Weaviate Docker container
# 2. Install Neo4j 5.15+
# 3. Configure credentials
```

---

### **3. Missing System Dependencies (HIGH)**

**Problem:** Required OCR and PDF processing tools are not installed.

**Required Tools:**
- **Tesseract OCR** - For text extraction from scanned PDFs
- **Ghostscript** - For Camelot table extraction
- **Poppler-utils** - For PDF rendering

**Impact:** 
- Table extraction will fail on scanned documents
- Camelot will not work
- OCR features disabled

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr ghostscript poppler-utils

# macOS
brew install tesseract ghostscript poppler

# Windows
# Download and install from official websites
```

---

### **4. Missing Python Dependencies (HIGH)**

**Problem:** Some critical packages may not be installed.

**Required Packages:**
- `weaviate-client` - Weaviate Python client (REPLACES psycopg2-binary)
- `neo4j` - Neo4j Python driver
- `sentence-transformers` - Embedding models
- `camelot-py[cv]` - Table extraction
- `layoutparser` - Document layout analysis
- `transformers` - NLP models
- `easyocr` - Advanced OCR

**Impact:** Pipeline will fail or skip critical stages.

**Solution:**
```bash
# Install all requirements
pip install -r requirements.txt

# Key Weaviate migration packages:
pip install weaviate-client==4.5.1 sentence-transformers
pip install neo4j camelot-py[cv] pytesseract layoutparser
pip install easyocr transformers fuzzywuzzy python-levenshtein

# Remove old PostgreSQL dependency
pip uninstall psycopg2-binary
```

---

### **5. Configuration Files Missing (MEDIUM)**

**Problem:** No environment configuration files exist.

**Expected Files:**
- `.env` - Database credentials, API keys
- `.env.example` - Template (exists in README but may not be in repo)

**Impact:** Hardcoded credentials may not match your setup.

**Solution:**
```bash
# Create .env file for Weaviate
cat > .env << 'EOF'
# Weaviate Configuration (NEW)
WEAVIATE_URL=http://localhost:8080
WEAVIATE_TIMEOUT=30
WEAVIATE_BATCH_SIZE=100

# Neo4j Configuration (UNCHANGED)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

# Model Configuration (UNCHANGED)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
SIMILARITY_THRESHOLD=0.7
MAX_RESULTS=10
EOF
```

---

### **6. Directory Structure Not Initialized (LOW)**

**Problem:** Required directories may not exist.

**Expected Structure:**
```
data/
├── raw/                    # Raw scraped data
├── preprocessed/
│   └── documents/          # PDF files ready for processing
├── extracted/              # Stage 1 output
├── normalized/             # Stage 2 output
├── weaviate/               # Stage 3 output (CHANGED from bridge_table)
├── neo4j/                  # Stage 4 output
└── embeddings/             # Stage 5 output (optional with Weaviate)
```

**Solution:**
```bash
# Updated directory structure for Weaviate
mkdir -p data/{raw,preprocessed/documents,extracted,normalized,weaviate,neo4j,embeddings}
mkdir -p logs cache outputs
```

---

## 📋 Recommended Action Plan

### **Phase 1: Environment Setup (1-2 hours) - UPDATED FOR WEAVIATE**

1. **Install System Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y docker.io docker-compose
   sudo apt-get install -y tesseract-ocr ghostscript poppler-utils
   
   # macOS
   brew install docker tesseract ghostscript poppler
   ```

2. **Install Python Dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   
   # Remove old PostgreSQL dependencies
   pip uninstall psycopg2-binary || true
   
   # Install Weaviate and required packages
   pip install weaviate-client==4.5.1 sentence-transformers
   pip install neo4j fastapi uvicorn streamlit
   pip install -r requirements.txt
   ```

3. **Create Directory Structure**
   ```bash
   # Updated for Weaviate
   mkdir -p data/{raw,preprocessed/documents,extracted,normalized,weaviate,neo4j,embeddings}
   mkdir -p logs cache outputs
   ```

4. **Start Services and Configure Databases**
   ```bash
   # Start Weaviate and Neo4j with Docker Compose
   docker-compose up -d
   
   # Wait for services to start
   sleep 30
   
   # Run setup script (updated for Weaviate)
   python pipeline/utils/setup_database.py
   
   # Test connections
   python pipeline/utils/setup_database.py --test-only
   ```

---

### **Phase 2: Data Acquisition (2-4 hours)**

1. **Option A: Manual Download (Recommended for POC)**
   - Download 5-10 key PDFs manually from:
     - `goir.ap.gov.in` (Government Orders)
     - `cse.ap.gov.in` (RTE documents)
     - `udiseplus.gov.in` (UDISE+ reports)
   - Place in `data/preprocessed/documents/`

2. **Option B: Run Scrapers (May Fail)**
   ```bash
   python data_pipeline/scrapers/go_scraper.py
   python data_pipeline/scrapers/cse_scraper.py
   python data_pipeline/scrapers/scert_scraper.py
   ```

3. **Verify Data**
   ```bash
   ls -lh data/preprocessed/documents/
   # Should show PDF files
   ```

---

### **Phase 3: Pipeline Execution (30-60 minutes per stage) - UPDATED FOR WEAVIATE**

1. **Test Database Connections**
   ```bash
   python pipeline/utils/setup_database.py --test-only
   
   # Should show:
   # ✅ Weaviate connection: OK
   # ✅ Neo4j connection: OK
   ```

2. **Run Pipeline Stage by Stage**
   ```bash
   # Stage 1: Extract tables (UNCHANGED)
   python pipeline/stages/1_extract_tables.py
   
   # Verify output
   ls -la data/extracted/
   cat data/extracted/all_extracted_data.json | jq length
   
   # Stage 2: Normalize schema (UNCHANGED)
   python pipeline/stages/2_normalize_schema.py
   
   # Stage 3: Load Weaviate (CHANGED - replaces PostgreSQL)
   python pipeline/stages/3_build_fact_table.py
   
   # Verify Weaviate loading
   ls -la data/weaviate/
   curl http://localhost:8080/v1/objects/Fact | jq length
   
   # Stage 4: Load Neo4j (UNCHANGED)
   python pipeline/stages/4_load_neo4j.py
   
   # Stage 5: Verify Weaviate indexes (SIMPLIFIED)
   python pipeline/stages/5_verify_weaviate.py
   ```

3. **Validate Each Stage**
   ```bash
   python pipeline/validation/validate_pipeline.py --stage extraction
   python pipeline/validation/validate_pipeline.py --stage normalization
   python pipeline/validation/validate_pipeline.py --stage weaviate
   # etc.
   ```

---

### **Phase 4: Debugging (Ongoing)**

**Common Error Patterns:**

1. **"No such file or directory"**
   - Missing input data
   - Incorrect paths
   - Directories not created

2. **"Connection refused"**
   - Weaviate/Neo4j not running
   - Wrong port/credentials (Weaviate: 8080, Neo4j: 7687)
   - Docker services not started
   - Firewall blocking connection

3. **"ModuleNotFoundError"**
   - Missing Python package
   - Wrong virtual environment
   - Requirements not installed
   - Old psycopg2-binary package conflicts

4. **"Extraction failed"**
   - PDF corrupt/encrypted
   - OCR not working
   - Camelot dependencies missing

**Debug Commands:**
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python pipeline/run_pipeline.py

# Check logs
tail -f logs/pipeline.log
tail -f extraction.log

# Test individual components (UPDATED FOR WEAVIATE)
python -c "import weaviate; print('Weaviate client OK')"
python -c "import neo4j; print('Neo4j OK')"
python -c "import camelot; print('Camelot OK')"

# Test Weaviate connection
curl http://localhost:8080/v1/.well-known/ready
curl http://localhost:8080/v1/schema

# Test Docker services
docker ps | grep -E "(weaviate|neo4j)"
docker logs ap_policy_weaviate
docker logs ap_policy_neo4j
```

---

## 🎯 Quick Start (Minimal POC) - WEAVIATE VERSION

If you just want to test the system quickly:

```bash
# 1. Setup (UPDATED FOR WEAVIATE)
pip install weaviate-client==4.5.1 sentence-transformers pandas numpy
docker-compose up -d weaviate
sleep 10  # Wait for Weaviate to start
python pipeline/utils/setup_database.py --weaviate-only

# 2. Get minimal data (2-3 PDFs)
mkdir -p data/preprocessed/documents
# Manually download 2-3 PDFs to this folder

# 3. Run first 3 stages (including Weaviate loading)
python pipeline/stages/1_extract_tables.py
python pipeline/stages/2_normalize_schema.py
python pipeline/stages/3_build_fact_table.py

# 4. Check outputs
ls -la data/extracted/
ls -la data/normalized/
ls -la data/weaviate/
curl http://localhost:8080/v1/objects/Fact | jq length
```

---

## 📊 Expected Pipeline Flow - UPDATED FOR WEAVIATE

```
Raw PDFs (data/preprocessed/documents/)
    ↓
[Stage 1] Extract Tables & Text
    ↓ data/extracted/all_extracted_data.json
[Stage 2] Normalize Schema
    ↓ data/normalized/normalized_facts.json
[Stage 3] Load Weaviate Vector Database (CHANGED)
    ↓ Weaviate database + data/weaviate/
[Stage 4] Load Neo4j Graph (UNCHANGED)
    ↓ Neo4j database + data/neo4j/
[Stage 5] Verify Weaviate Indexes (SIMPLIFIED)
    ↓ Weaviate collection verification
[Stage 6] Start RAG API (UNCHANGED)
    ↓ FastAPI server (localhost:8000)
[Stage 7] Launch Dashboard (UNCHANGED)
    ↓ Streamlit UI (localhost:8501)
```

**Key Changes:**
- **Stage 3**: PostgreSQL+pgvector → Weaviate (with built-in vector indexing)
- **Stage 5**: Complex pgvector indexing → Simple verification
- **All stages**: Embedding generation handled by Weaviate client

---

## 🔍 Diagnostic Commands

```bash
# Check directory structure
tree -L 3 data/

# Count files at each stage
echo "PDFs: $(ls data/preprocessed/documents/*.pdf 2>/dev/null | wc -l)"
echo "Extracted: $(ls data/extracted/*.json 2>/dev/null | wc -l)"
echo "Normalized: $(ls data/normalized/*.json 2>/dev/null | wc -l)"

# Check database connectivity (UPDATED FOR WEAVIATE)
curl http://localhost:8080/v1/.well-known/ready  # Weaviate health
curl http://localhost:8080/v1/schema             # Weaviate schema
curl http://localhost:7474                       # Neo4j browser

# Check Python environment (UPDATED)
python --version
pip list | grep -E "weaviate|neo4j|sentence|camelot|torch"

# System dependencies
tesseract --version
gs --version
pdftotext -v
```

---

## 🚧 Known Limitations

1. **Scrapers may be outdated** - Government websites change frequently
2. **OCR quality depends on PDF** - Scanned documents need good resolution
3. **Table extraction is fragile** - Complex tables may not parse correctly
4. **No error recovery** - Pipeline stops on first failure
5. **No resume capability** - Must restart from beginning if interrupted
6. **No parallel processing** - Processes one PDF at a time
7. **Hardcoded thresholds** - Fuzzy matching may miss variations

---

## ✅ Success Criteria - UPDATED FOR WEAVIATE

You'll know the pipeline is working when:

1. ✅ All directories contain files
2. ✅ `all_extracted_data.json` has >0 items  
3. ✅ `normalized_facts.json` has >0 facts
4. ✅ **Weaviate `Fact` collection has >0 objects** (CHANGED)
5. ✅ Neo4j has `Fact`, `Indicator`, `District` nodes
6. ✅ API responds at `http://localhost:8000/health`
7. ✅ Dashboard loads at `http://localhost:8501`

**Verification Commands:**
```bash
# Check Weaviate data (REPLACES PostgreSQL check)
curl http://localhost:8080/v1/objects/Fact | jq length

# Check Neo4j data (UNCHANGED)
curl -u neo4j:password http://localhost:7474/db/data/

# Check API health
curl http://localhost:8000/health

# Test search functionality
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "enrollment statistics", "limit": 5}'
```

---

## 📞 Next Steps

**Immediate actions:**
1. Run diagnostic commands above
2. Share the output with me
3. I'll provide specific fixes based on actual errors

**Questions to answer:**
- Do you have Docker and Weaviate/Neo4j running?
- Do you have any PDF files downloaded?
- What errors are you seeing in the logs?
- Which stage is failing?

---

## 📝 Summary - MIGRATED TO WEAVIATE

**Previous architecture:** PostgreSQL + pgvector + Neo4j + Custom RAG
**New architecture:** **Weaviate + Neo4j + Simplified RAG**

**Key improvements:**
- ✅ **Simpler setup** - Docker Compose handles everything
- ✅ **Better search** - Built-in hybrid search (vector + BM25)
- ✅ **Faster development** - No complex pgvector configuration
- ✅ **Production ready** - Cloud-native vector database

**Root cause:** The pipeline expects a fully configured environment with Weaviate and Neo4j running, PDFs downloaded, and all dependencies installed.

**Quick fix:** Follow the updated Phase 1 (Weaviate Environment Setup) and Phase 2 (Data Acquisition) above.

**Long-term fix:** The Docker Compose setup already handles most prerequisites automatically.

---

## 🚀 Weaviate Migration Benefits

### **Why We Migrated from PostgreSQL+pgvector to Weaviate:**

1. **Setup Complexity** 
   - **Before:** Install PostgreSQL → Add pgvector extension → Configure vector columns
   - **After:** `docker-compose up -d weaviate` 

2. **Search Quality**
   - **Before:** Custom vector similarity + manual BM25 implementation
   - **After:** Built-in hybrid search with configurable alpha parameter

3. **Developer Experience**
   - **Before:** Complex SQL queries + manual embedding management
   - **After:** Modern Python client with automatic vectorization

4. **Performance**
   - **Before:** B-tree indexes for vectors (suboptimal)
   - **After:** HNSW algorithm purpose-built for vectors

5. **Scalability**
   - **Before:** Vertical scaling only
   - **After:** Horizontal scaling + cloud deployment ready

Next actions:
1. **Start with infrastructure changes** (Docker Compose, environment variables)
2. **Update pipeline stages** (Stage 3 database loading)
3. **Update backend API** (retriever and main.py)
4. **Test end-to-end** with sample data
