#!/bin/bash
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
    print(f'Total Processed: {data["processed_count"]}/{data["total_expected"]}')
    print(f'Categories: {data["categories"]}')
else:
    print('Metadata file not found')
"

echo "✅ Batch processing complete!"
echo "🌐 View results at: http://localhost:7474"
