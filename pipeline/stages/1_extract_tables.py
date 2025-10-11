#!/usr/bin/env python3
"""
Stage 1: Table Extraction Engine
Extracts structured tables and textual policy statements from PDFs with high fidelity
"""
import os
import json
import logging
import io
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# PDF Processing
import fitz  # PyMuPDF
import camelot
from PIL import Image
import cv2
import numpy as np

# OCR and Layout Analysis
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False

# Validation
import pandera as pa
from pandera import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)

class TableExtractionEngine:
    """Production-ready table extraction engine for AP education policy documents"""
    
    def __init__(self, output_dir: str = "data/extracted"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize layout parser if available
        if LAYOUTPARSER_AVAILABLE:
            try:
                self.layout_model = lp.Detectron2LayoutModel(
                    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                )
            except Exception as e:
                logger.warning(f"LayoutParser model loading failed: {e}")
                self.layout_model = None
        else:
            self.layout_model = None
            logger.warning("LayoutParser not available. OCR functionality will be limited.")
        
        # Andhra Pradesh districts (canonical list)
        self.ap_districts = [
            'Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Kadapa', 
            'Krishna', 'Kurnool', 'Nellore', 'Prakasam', 'Srikakulam', 
            'Visakhapatnam', 'Vizianagaram', 'West Godavari'
        ]
        
        # Common education indicators
        self.education_indicators = [
            'GER', 'NER', 'GPI', 'PTR', 'Dropout Rate', 'Retention Rate',
            'Enrolment', 'Teachers', 'Schools', 'Classrooms', 'Toilets',
            'Drinking Water', 'Electricity', 'Computer', 'Library'
        ]
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract all tables and textual data from a PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted table/text objects
        """
        logger.info(f"🔍 EXTRACTING DATA FROM: {pdf_path}")
        
        extracted_data = []
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info(f"📄 PDF opened successfully - {total_pages} pages detected")
        
        try:
            for page_num in range(total_pages):
                page = doc[page_num]
                logger.info(f"📖 Processing page {page_num + 1}/{total_pages}")
                
                # Extract text tables using Camelot
                logger.info(f"🔧 Extracting text tables from page {page_num + 1} using Camelot...")
                camelot_tables = self._extract_camelot_tables(pdf_path, page_num)
                logger.info(f"✅ Found {len(camelot_tables)} text tables on page {page_num + 1}")
                extracted_data.extend(camelot_tables)
                
                # Extract image tables using OCR
                if self.layout_model and TESSERACT_AVAILABLE:
                    logger.info(f"🔧 Extracting image tables from page {page_num + 1} using OCR...")
                    ocr_tables = self._extract_ocr_tables(page, page_num)
                    logger.info(f"✅ Found {len(ocr_tables)} image tables on page {page_num + 1}")
                    extracted_data.extend(ocr_tables)
                else:
                    logger.warning(f"⚠️ OCR extraction skipped - LayoutParser or Tesseract not available")
                
                # Extract policy text paragraphs
                logger.info(f"🔧 Extracting policy text from page {page_num + 1}...")
                policy_texts = self._extract_policy_text(page, page_num)
                logger.info(f"✅ Found {len(policy_texts)} policy text segments on page {page_num + 1}")
                extracted_data.extend(policy_texts)
        
        finally:
            doc.close()
        
        logger.info(f"📊 RAW EXTRACTION COMPLETE: {len(extracted_data)} total items extracted")
        
        # Validate and clean extracted data
        logger.info(f"🔧 Validating and cleaning extracted data...")
        validated_data = self._validate_extracted_data(extracted_data, pdf_path)
        logger.info(f"✅ Validation complete: {len(validated_data)} items passed validation")
        
        # Save extracted data
        logger.info(f"💾 Saving validated data...")
        self._save_extracted_data(validated_data, pdf_path)
        
        logger.info(f"✅ EXTRACTION COMPLETE: {len(validated_data)} validated items from {pdf_path}")
        logger.info(f"📊 Breakdown by type:")
        
        # Count by type
        type_counts = {}
        for item in validated_data:
            item_type = item.get('type', 'unknown')
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        for item_type, count in type_counts.items():
            logger.info(f"   - {item_type}: {count} items")
        
        return validated_data
    
    def _extract_camelot_tables(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables using Camelot (stream method)"""
        tables = []
        
        try:
            logger.info(f"🔧 Camelot: Extracting tables from page {page_num + 1}...")
            # Extract tables from specific page
            camelot_tables = camelot.read_pdf(
                pdf_path, 
                pages=str(page_num + 1), 
                flavor='stream',
                edge_tol=500,
                row_tol=10
            )
            
            logger.info(f"🔍 Camelot: Found {len(camelot_tables)} potential tables on page {page_num + 1}")
            
            for i, table in enumerate(camelot_tables):
                logger.info(f"📊 Processing table {i + 1}/{len(camelot_tables)} - Shape: {table.df.shape}, Accuracy: {table.accuracy:.2f}")
                
                table_data = {
                    'doc_id': Path(pdf_path).stem,
                    'page': page_num + 1,
                    'table_id': f"T_{page_num + 1:03d}_{i:02d}",
                    'extraction_method': 'camelot_stream',
                    'caption': self._extract_table_caption(table.df),
                    'headers': self._extract_headers(table.df),
                    'rows': table.df.values.tolist(),
                    'shape': table.df.shape,
                    'confidence': table.accuracy,
                    'source_type': self._classify_source_type(pdf_path),
                    'year': self._extract_year(pdf_path),
                    'extracted_at': datetime.now().isoformat()
                }
                
                logger.info(f"📋 Table {i + 1} caption: '{table_data['caption']}'")
                logger.info(f"📋 Table {i + 1} headers: {table_data['headers']}")
                
                # Validate table structure
                if self._validate_table_structure(table_data):
                    logger.info(f"✅ Table {i + 1} passed validation")
                    tables.append(table_data)
                else:
                    logger.warning(f"⚠️ Table {i + 1} failed validation - skipping")
        
        except Exception as e:
            logger.error(f"❌ Camelot extraction failed for page {page_num}: {e}")
        
        return tables
    
    def _extract_ocr_tables(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables from images using OCR and layout analysis"""
        tables = []
        
        try:
            logger.info(f"🔧 OCR: Converting page {page_num + 1} to image...")
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            logger.info(f"📸 Image converted - Size: {img.size}")
            
            # Detect layout
            logger.info(f"🔍 OCR: Detecting layout on page {page_num + 1}...")
            layout = self.layout_model.detect(img_array)
            
            # Find table regions
            table_regions = [block for block in layout if block.type == 'Table']
            logger.info(f"🔍 OCR: Found {len(table_regions)} table regions on page {page_num + 1}")
            
            for i, region in enumerate(table_regions):
                logger.info(f"📊 OCR: Processing table region {i + 1}/{len(table_regions)}")
                # Crop table region
                x1, y1, x2, y2 = region.coordinates
                logger.info(f"📐 Table region coordinates: ({x1}, {y1}, {x2}, {y2})")
                table_img = img.crop((x1, y1, x2, y2))
                
                # OCR the table
                logger.info(f"🔤 OCR: Extracting text from table region {i + 1}...")
                table_text = pytesseract.image_to_string(table_img, config='--psm 6')
                logger.info(f"📝 OCR text length: {len(table_text)} characters")
                
                # Parse table structure
                logger.info(f"🔧 OCR: Parsing table structure for region {i + 1}...")
                table_data = self._parse_ocr_table(table_text, page_num, i)
                if table_data:
                    logger.info(f"✅ OCR: Successfully parsed table {i + 1}")
                    tables.append(table_data)
                else:
                    logger.warning(f"⚠️ OCR: Failed to parse table {i + 1}")
        
        except Exception as e:
            logger.error(f"❌ OCR table extraction failed for page {page_num}: {e}")
        
        logger.info(f"✅ OCR extraction complete: {len(tables)} tables extracted from page {page_num + 1}")
        return tables
    
    def _extract_policy_text(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Extract policy text paragraphs"""
        texts = []
        
        try:
            logger.info(f"🔧 Text: Extracting policy text from page {page_num + 1}...")
            # Get text blocks
            text_dict = page.get_text("dict")
            
            total_blocks = len(text_dict["blocks"])
            logger.info(f"📄 Found {total_blocks} text blocks on page {page_num + 1}")
            
            policy_blocks = 0
            for block_idx, block in enumerate(text_dict["blocks"]):
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                    
                    # Check if it's policy-related text
                    if self._is_policy_text(block_text):
                        policy_blocks += 1
                        logger.info(f"📝 Policy text block {policy_blocks}: {len(block_text.strip())} chars")
                        
                        text_data = {
                            'doc_id': Path(page.parent.name).stem,
                            'page': page_num + 1,
                            'text_id': f"TXT_{page_num + 1:03d}_{len(texts):02d}",
                            'extraction_method': 'pymupdf_text',
                            'text': block_text.strip(),
                            'source_type': self._classify_source_type(page.parent.name),
                            'year': self._extract_year(page.parent.name),
                            'extracted_at': datetime.now().isoformat()
                        }
                        texts.append(text_data)
            
            logger.info(f"✅ Text extraction complete: {len(texts)} policy text blocks from {total_blocks} total blocks")
        
        except Exception as e:
            logger.error(f"❌ Policy text extraction failed for page {page_num}: {e}")
        
        return texts
    
    def _extract_table_caption(self, df: pd.DataFrame) -> str:
        """Extract table caption from DataFrame"""
        # Look for caption-like text in first few rows
        for i in range(min(3, len(df))):
            row_text = ' '.join(str(cell) for cell in df.iloc[i].values if pd.notna(cell))
            if any(keyword in row_text.lower() for keyword in ['table', 'figure', 'enrolment', 'statistics']):
                return row_text
        return "Untitled Table"
    
    def _extract_headers(self, df: pd.DataFrame) -> List[str]:
        """Extract and clean table headers"""
        if len(df) == 0:
            return []
        
        # Use first row as headers
        headers = [str(col) for col in df.columns]
        
        # Clean headers
        cleaned_headers = []
        for header in headers:
            # Remove common artifacts
            cleaned = header.replace('\n', ' ').replace('\r', ' ').strip()
            cleaned = ' '.join(cleaned.split())  # Normalize whitespace
            cleaned_headers.append(cleaned)
        
        return cleaned_headers
    
    def _classify_source_type(self, pdf_path: str) -> str:
        """Classify document source type"""
        path_lower = str(pdf_path).lower()
        
        if 'cse' in path_lower:
            return 'CSE'
        elif 'scert' in path_lower:
            return 'SCERT'
        elif 'go' in path_lower:
            return 'GO'
        else:
            return 'UNKNOWN'
    
    def _extract_year(self, pdf_path: str) -> str:
        """Extract year from PDF filename or content"""
        import re
        
        # Try to extract year from filename
        filename = Path(pdf_path).name
        year_match = re.search(r'(\d{4})', filename)
        if year_match:
            return year_match.group(1)
        
        # Default to current year if not found
        return str(datetime.now().year)
    
    def _validate_table_structure(self, table_data: Dict[str, Any]) -> bool:
        """Validate table structure and content"""
        try:
            # Check minimum size
            if table_data['shape'][0] < 2 or table_data['shape'][1] < 2:
                return False
            
            # Check for numeric data
            rows = table_data['rows']
            numeric_cells = 0
            total_cells = 0
            
            for row in rows[1:]:  # Skip header
                for cell in row:
                    total_cells += 1
                    if isinstance(cell, (int, float)) or (isinstance(cell, str) and cell.replace('.', '').replace(',', '').isdigit()):
                        numeric_cells += 1
            
            # At least 30% numeric data
            if total_cells > 0 and numeric_cells / total_cells < 0.3:
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Table validation failed: {e}")
            return False
    
    def _is_policy_text(self, text: str) -> bool:
        """Check if text contains policy-related content"""
        policy_keywords = [
            'policy', 'guideline', 'directive', 'instruction', 'order',
            'education', 'school', 'teacher', 'student', 'curriculum',
            'enrolment', 'dropout', 'retention', 'assessment', 'evaluation'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in policy_keywords) and len(text.strip()) > 50
    
    def _parse_ocr_table(self, ocr_text: str, page_num: int, table_idx: int) -> Optional[Dict[str, Any]]:
        """Parse OCR text into table structure"""
        try:
            lines = ocr_text.strip().split('\n')
            if len(lines) < 2:
                return None
            
            # Simple parsing - split by whitespace
            rows = []
            for line in lines:
                if line.strip():
                    cells = line.split()
                    if len(cells) >= 2:  # At least 2 columns
                        rows.append(cells)
            
            if len(rows) < 2:
                return None
            
            table_data = {
                'doc_id': 'OCR_EXTRACTED',
                'page': page_num + 1,
                'table_id': f"OCR_{page_num + 1:03d}_{table_idx:02d}",
                'extraction_method': 'ocr_tesseract',
                'caption': 'OCR Extracted Table',
                'headers': rows[0] if rows else [],
                'rows': rows[1:] if len(rows) > 1 else [],
                'shape': (len(rows) - 1, len(rows[0])) if rows else (0, 0),
                'confidence': 0.7,  # Default confidence for OCR
                'source_type': 'OCR',
                'year': str(datetime.now().year),
                'extracted_at': datetime.now().isoformat()
            }
            
            return table_data
        
        except Exception as e:
            logger.error(f"OCR table parsing failed: {e}")
            return None
    
    def _validate_extracted_data(self, extracted_data: List[Dict[str, Any]], pdf_path: str) -> List[Dict[str, Any]]:
        """Validate and clean extracted data"""
        logger.info(f"🔧 VALIDATION: Starting validation of {len(extracted_data)} items...")
        validated_data = []
        validation_stats = {
            'total': len(extracted_data),
            'passed': 0,
            'failed': 0,
            'missing_fields': 0,
            'validation_errors': 0
        }
        
        for item_idx, item in enumerate(extracted_data):
            try:
                logger.info(f"🔍 Validating item {item_idx + 1}/{len(extracted_data)}: {item.get('table_id', item.get('text_id', 'unknown'))}")
                
                # Add checksum for data integrity
                item['checksum'] = self._calculate_checksum(item)
                logger.info(f"✅ Checksum calculated: {item['checksum'][:8]}...")
                
                # Validate required fields
                required_fields = ['doc_id', 'page', 'extraction_method', 'source_type', 'year']
                missing_fields = [field for field in required_fields if field not in item]
                
                if not missing_fields:
                    validated_data.append(item)
                    validation_stats['passed'] += 1
                    logger.info(f"✅ Item {item_idx + 1} passed validation")
                else:
                    validation_stats['missing_fields'] += 1
                    logger.warning(f"⚠️ Item {item_idx + 1} missing fields: {missing_fields}")
            
            except Exception as e:
                validation_stats['validation_errors'] += 1
                logger.error(f"❌ Validation failed for item {item_idx + 1}: {e}")
                continue
        
        validation_stats['failed'] = validation_stats['total'] - validation_stats['passed']
        
        logger.info(f"📊 VALIDATION COMPLETE:")
        logger.info(f"   - Total items: {validation_stats['total']}")
        logger.info(f"   - Passed: {validation_stats['passed']}")
        logger.info(f"   - Failed: {validation_stats['failed']}")
        logger.info(f"   - Missing fields: {validation_stats['missing_fields']}")
        logger.info(f"   - Validation errors: {validation_stats['validation_errors']}")
        
        return validated_data
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity"""
        import hashlib
        
        # Create a stable string representation
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _save_extracted_data(self, extracted_data: List[Dict[str, Any]], pdf_path: str):
        """Save extracted data to JSON file"""
        try:
            output_file = self.output_dir / f"{Path(pdf_path).stem}_extracted.json"
            logger.info(f"💾 SAVING: Writing {len(extracted_data)} items to {output_file}")
            
            # Calculate file size before writing
            data_size = len(json.dumps(extracted_data, indent=2, ensure_ascii=False))
            logger.info(f"📊 Data size: {data_size:,} bytes ({data_size/1024:.1f} KB)")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
            
            # Verify file was written
            actual_size = output_file.stat().st_size
            logger.info(f"✅ File saved successfully - Size: {actual_size:,} bytes")
            
            # Log breakdown by type
            type_counts = {}
            for item in extracted_data:
                item_type = item.get('type', 'unknown')
                type_counts[item_type] = type_counts.get(item_type, 0) + 1
            
            logger.info(f"📊 Saved data breakdown:")
            for item_type, count in type_counts.items():
                logger.info(f"   - {item_type}: {count} items")
        
        except Exception as e:
            logger.error(f"❌ Failed to save extracted data: {e}")
    
    def process_all_pdfs(self, pdf_directory: str = "data/preprocessed/documents") -> Dict[str, List[Dict[str, Any]]]:
        """Process all PDFs in directory"""
        logger.info(f"🚀 BATCH PROCESSING: Starting extraction from {pdf_directory}")
        
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            logger.error(f"❌ PDF directory not found: {pdf_directory}")
            return {}
        
        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"📁 Found {len(pdf_files)} PDF files to process")
        
        if not pdf_files:
            logger.warning(f"⚠️ No PDF files found in {pdf_directory}")
            return {}
        
        all_extracted_data = {}
        processing_stats = {
            'total': len(pdf_files),
            'processed': 0,
            'failed': 0,
            'total_items': 0
        }
        
        for pdf_idx, pdf_file in enumerate(pdf_files):
            try:
                logger.info(f"📄 PROCESSING PDF {pdf_idx + 1}/{len(pdf_files)}: {pdf_file.name}")
                start_time = time.time()
                
                extracted_data = self.extract_from_pdf(str(pdf_file))
                all_extracted_data[pdf_file.name] = extracted_data
                
                processing_time = time.time() - start_time
                processing_stats['processed'] += 1
                processing_stats['total_items'] += len(extracted_data)
                
                logger.info(f"✅ PDF {pdf_idx + 1} completed in {processing_time:.2f}s - {len(extracted_data)} items extracted")
                
                # Rate limiting
                logger.info(f"⏱️ Rate limiting: waiting 1 second...")
                time.sleep(1)
            
            except Exception as e:
                processing_stats['failed'] += 1
                logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
                continue
        
        # Save combined results
        logger.info(f"💾 SAVING COMBINED RESULTS...")
        combined_file = self.output_dir / "all_extracted_data.json"
        
        combined_size = len(json.dumps(all_extracted_data, indent=2, ensure_ascii=False))
        logger.info(f"📊 Combined data size: {combined_size:,} bytes ({combined_size/1024/1024:.1f} MB)")
        
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_extracted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Combined results saved to {combined_file}")
        
        # Final statistics
        logger.info(f"📊 BATCH PROCESSING COMPLETE:")
        logger.info(f"   - Total PDFs: {processing_stats['total']}")
        logger.info(f"   - Successfully processed: {processing_stats['processed']}")
        logger.info(f"   - Failed: {processing_stats['failed']}")
        logger.info(f"   - Total items extracted: {processing_stats['total_items']}")
        logger.info(f"   - Average items per PDF: {processing_stats['total_items']/max(processing_stats['processed'], 1):.1f}")
        
        return all_extracted_data

def main():
    """Main function to run table extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract tables from PDFs')
    parser.add_argument('--pdf-dir', default='data/preprocessed/documents',
                       help='Directory containing PDF files')
    parser.add_argument('--output-dir', default='data/extracted',
                       help='Output directory for extracted data')
    
    args = parser.parse_args()
    
    # Setup logging with detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('extraction.log')
        ]
    )
    
    logger.info("🚀 STARTING TABLE EXTRACTION PIPELINE")
    logger.info(f"📁 PDF Directory: {args.pdf_dir}")
    logger.info(f"📁 Output Directory: {args.output_dir}")
    
    # Initialize extraction engine
    logger.info("🔧 Initializing Table Extraction Engine...")
    engine = TableExtractionEngine(output_dir=args.output_dir)
    logger.info("✅ Extraction Engine initialized successfully")
    
    # Process all PDFs
    logger.info("🚀 Starting batch processing...")
    start_time = time.time()
    
    results = engine.process_all_pdfs(args.pdf_dir)
    
    total_time = time.time() - start_time
    
    # Print summary
    total_items = sum(len(items) for items in results.values())
    
    logger.info("🎉 EXTRACTION PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"📊 FINAL SUMMARY:")
    logger.info(f"   - PDFs processed: {len(results)}")
    logger.info(f"   - Total items extracted: {total_items}")
    logger.info(f"   - Processing time: {total_time:.2f} seconds")
    logger.info(f"   - Average time per PDF: {total_time/max(len(results), 1):.2f} seconds")
    logger.info(f"   - Output directory: {args.output_dir}")
    logger.info("=" * 60)
    
    # Print per-PDF breakdown
    logger.info("📋 PER-PDF BREAKDOWN:")
    for pdf_name, items in results.items():
        logger.info(f"   - {pdf_name}: {len(items)} items")
    
    print(f"\n🎉 Extraction Complete!")
    print(f"📊 PDFs processed: {len(results)}")
    print(f"📊 Total items extracted: {total_items}")
    print(f"⏱️ Processing time: {total_time:.2f} seconds")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📄 Detailed logs saved to: extraction.log")

if __name__ == "__main__":
    main()
