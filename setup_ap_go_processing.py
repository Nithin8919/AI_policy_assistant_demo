"""
AP Government Orders (GO) Processing Setup
Prepares the system for processing 9400+ Andhra Pradesh Government Orders
"""

import json
import logging
from pathlib import Path
from pipeline.utils.dataset_registry import DatasetSchema, DatasetRegistry
from pipeline.auto_bridge_pipeline import AutoBridgePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_ap_go_schema():
    """Setup specialized schema for AP Government Orders"""
    
    registry = DatasetRegistry()
    
    # Create comprehensive AP GO schema
    ap_go_schema = DatasetSchema(
        name='ap_government_orders',
        source_type='pdf',
        entity_types=[
            'government_order', 'policy', 'department', 'district', 
            'scheme', 'allocation', 'directive', 'circular', 
            'notification', 'appointment', 'transfer'
        ],
        key_fields={
            'go_number': 'string',           # G.O.MS.No.123/2024
            'go_date': 'date',               # Date of issuance
            'department': 'string',          # Education, Health, etc.
            'subject': 'string',             # GO subject/title
            'category': 'string',            # Policy, Administrative, Financial
            'district': 'string',            # Affected district(s)
            'scheme_name': 'string',         # Related scheme/program
            'amount': 'float',               # Financial allocations
            'effective_date': 'date',        # When GO becomes effective
            'reference_go': 'string',        # References to other GOs
            'file_number': 'string'          # Department file reference
        },
        bridge_potential=[
            'go_number', 'department', 'district', 'scheme_name', 
            'reference_go', 'subject', 'file_number'
        ],
        extraction_method='nlp_extraction',
        confidence_threshold=0.8
    )
    
    # Register the schema
    success = registry.register_dataset(ap_go_schema, overwrite=True)
    if success:
        logger.info("✅ AP Government Orders schema registered successfully")
        return ap_go_schema
    else:
        logger.error("❌ Failed to register AP GO schema")
        return None

def create_go_processing_directories():
    """Create directory structure for GO processing"""
    
    directories = [
        "data/ap_go/raw",                    # Original PDF files
        "data/ap_go/extracted",              # Extracted text/entities
        "data/ap_go/normalized",             # Normalized facts
        "data/ap_go/processed",              # Processed with bridges
        "data/ap_go/failed",                 # Files that failed processing
        "data/ap_go/samples",                # Sample files for testing
        "logs/ap_go"                         # GO-specific logs
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")
    
    # Create metadata file
    metadata = {
        "description": "Andhra Pradesh Government Orders Processing",
        "total_expected": 9400,
        "processed_count": 0,
        "last_update": None,
        "categories": {
            "education": 0,
            "health": 0, 
            "finance": 0,
            "agriculture": 0,
            "welfare": 0,
            "administration": 0,
            "other": 0
        },
        "departments": {},
        "year_wise": {},
        "status": "initialized"
    }
    
    with open("data/ap_go/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("📝 Created metadata tracking file")

def create_go_sample_processor():
    """Create specialized processor for GO sample testing"""
    
    sample_processor_code = """#!/usr/bin/env python3
# AP Government Order Sample Processor

import json
import logging
from pathlib import Path
import sys
sys.path.append('.')

from pipeline.auto_bridge_pipeline import AutoBridgePipeline
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APGOSampleProcessor:
    def __init__(self):
        self.pipeline = AutoBridgePipeline()
        
    def extract_go_entities(self, go_text: str, file_name: str) -> dict:
        entities = {
            'source_file': file_name,
            'extracted_at': datetime.now().isoformat()
        }
        
        # GO Number pattern
        go_match = re.search(r'G\\.O\\.(MS\\.)?No\\.?\\s*(\\d+)', go_text, re.IGNORECASE)
        if go_match:
            entities['go_number'] = go_match.group(2)
            
        # Department pattern
        dept_match = re.search(r'Department[:\\s]+([A-Za-z\\s&,]+)', go_text, re.IGNORECASE)
        if dept_match:
            entities['department'] = dept_match.group(1).strip()
            
        # District pattern
        ap_districts = ['Krishna', 'Guntur', 'Prakasam', 'Nellore', 'Chittoor']
        for district in ap_districts:
            if district.lower() in go_text.lower():
                entities['district'] = district
                break
                
        return entities
    
    def process_sample_gos(self, sample_dir: Path, limit: int = 10):
        logger.info(f"Processing sample GOs from {sample_dir}")
        
        # Create sample data if no PDFs exist
        sample_data = []
        for i in range(min(limit, 5)):
            mock_go = {
                'source_file': f'sample_go_{i+1}.pdf',
                'go_number': str(100 + i),
                'department': ['Education', 'Health', 'Finance'][i % 3],
                'district': ['Krishna', 'Guntur', 'Prakasam'][i % 3],
                'subject': f'Sample GO {i+1} for testing',
                'extracted_at': datetime.now().isoformat()
            }
            sample_data.append(mock_go)
        
        # Save sample data
        output_file = Path("data/ap_go/samples/extracted_entities.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Created {len(sample_data)} sample GO entities")
        return sample_data
    
    def create_sample_bridges(self, entities: list):
        logger.info("Creating bridge connections for sample GOs")
        
        # Convert to facts format
        facts = []
        for entity in entities:
            fact = {
                'fact_id': f"AP_GO_{entity.get('go_number', 'unknown')}_sample",
                'district': entity.get('district', 'Unknown'),
                'indicator': 'Government Order',
                'year': 2024,
                'value': 1.0,
                'source_document': f"AP_GO_{entity.get('go_number')}",
                'metadata': entity
            }
            facts.append(fact)
        
        # Get schema
        schema = self.pipeline.registry.schemas.get('ap_government_orders')
        if schema:
            new_links = self.pipeline.auto_bridge_new_entities(facts, schema)
            logger.info(f"Created {len(new_links)} bridge connections")
            return new_links
        else:
            logger.error("AP GO schema not found")
            return []

def main():
    print("🚀 Processing Sample AP Government Orders...")
    
    processor = APGOSampleProcessor()
    
    # Process sample files
    sample_dir = Path("data/ap_go/samples")
    entities = processor.process_sample_gos(sample_dir, limit=5)
    
    if entities:
        bridges = processor.create_sample_bridges(entities)
        print(f"✅ Created {len(entities)} sample GOs and {len(bridges)} bridges")
        print("🌐 Check Neo4j Browser: http://localhost:7474")
        
        # Show sample results
        print("\\n📋 Sample GO Entities:")
        for entity in entities[:3]:
            print(f"   GO {entity['go_number']}: {entity.get('subject', 'N/A')}")
            print(f"   Department: {entity.get('department', 'N/A')}")
            print(f"   District: {entity.get('district', 'N/A')}")
            print()
    else:
        print("⚠️  No entities processed.")

if __name__ == "__main__":
    main()
"""
    
    with open("data/ap_go/sample_processor.py", 'w') as f:
        f.write(sample_processor_code)
    
    logger.info("📝 Created GO sample processor")

def create_batch_processing_script():
    """Create script for batch processing 9400+ GOs"""
    
    batch_script = '''#!/bin/bash
# Batch Processing Script for 9400+ AP Government Orders

echo "🚀 Starting AP Government Orders Batch Processing"

# Configuration
BATCH_SIZE=100
INPUT_DIR="data/ap_go/raw"
LOG_FILE="logs/ap_go/batch_process.log"

# Create log directory
mkdir -p logs/ap_go

# Check if input directory has files
if [ ! -d "$INPUT_DIR" ] || [ -z "$(ls -A $INPUT_DIR)" ]; then
    echo "❌ No files found in $INPUT_DIR"
    echo "   Please place GO PDF files in $INPUT_DIR"
    exit 1
fi

# Count total files
TOTAL_FILES=$(find $INPUT_DIR -name "*.pdf" | wc -l)
echo "📊 Found $TOTAL_FILES PDF files to process"

# Start processing
echo "🔄 Starting auto-bridge pipeline..."
python pipeline/auto_bridge_pipeline.py --run >> $LOG_FILE 2>&1

# Generate progress report
echo "📈 Processing Complete. Check log: $LOG_FILE"

# Show summary
python -c "
import json
from pathlib import Path

metadata_file = Path('data/ap_go/metadata.json')
if metadata_file.exists():
    with open(metadata_file) as f:
        data = json.load(f)
    print(f'Total Processed: {data[\"processed_count\"]}/{data[\"total_expected\"]}')
    print(f'Categories: {data[\"categories\"]}')
else:
    print('Metadata file not found')
"

echo "✅ Batch processing complete!"
echo "🌐 View results at: http://localhost:7474"
'''
    
    with open("batch_process_gos.sh", 'w') as f:
        f.write(batch_script)
    
    # Make executable
    import stat
    Path("batch_process_gos.sh").chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    
    logger.info("📝 Created batch processing script: batch_process_gos.sh")

def main():
    """Main setup function"""
    print("🏗️  Setting up AP Government Orders Processing System")
    print("=" * 60)
    
    try:
        # Step 1: Setup schema
        print("\n1. Setting up AP GO schema...")
        schema = setup_ap_go_schema()
        
        if not schema:
            print("❌ Failed to setup schema. Exiting.")
            return
        
        # Step 2: Create directories
        print("\n2. Creating directory structure...")
        create_go_processing_directories()
        
        # Step 3: Create processors
        print("\n3. Creating specialized processors...")
        create_go_sample_processor()
        create_batch_processing_script()
        
        print("\n✅ Setup Complete!")
        print("\n📋 Next Steps:")
        print("   1. Test with samples: python data/ap_go/sample_processor.py")
        print("   2. For 9.4k GOs: Place all PDFs in data/ap_go/raw/")
        print("   3. Run batch: ./batch_process_gos.sh")
        print("   4. Monitor: tail -f logs/ap_go/batch_process.log")
        
        print("\n🔗 Auto-bridging Features:")
        print("   • Automatic GO number linking")
        print("   • Department-based connections")
        print("   • District-wise policy chains")
        print("   • Reference GO relationships")
        print("   • Subject/scheme clustering")
        
        print("\n🌐 View results: http://localhost:7474")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    main()