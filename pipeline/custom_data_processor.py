#!/usr/bin/env python3
"""
Custom Data Processor for AP Policy Co-Pilot
Processes extracted data into searchable facts for Weaviate
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class CustomDataProcessor:
    """Custom processor to extract meaningful facts from policy documents"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Education indicators patterns
        self.education_patterns = {
            'enrollment': r'(\d+(?:,\d{3})*)\s*(?:students?|children|pupils?)\s*(?:enrolled|admission)',
            'schools': r'(\d+(?:,\d{3})*)\s*schools?',
            'teachers': r'(\d+(?:,\d{3})*)\s*teachers?',
            'dropout': r'(\d+(?:\.\d+)?)\s*%?\s*dropout',
            'literacy': r'(\d+(?:\.\d+)?)\s*%?\s*literacy',
            'attendance': r'(\d+(?:\.\d+)?)\s*%?\s*attendance',
            'infrastructure': r'(\d+(?:,\d{3})*)\s*(?:classrooms?|buildings?|toilets?)',
            'budget': r'(?:Rs\.?\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:crores?|lakhs?|thousand)?'
        }
        
        # AP districts
        self.ap_districts = [
            'Anantapur', 'Chittoor', 'East Godavari', 'Guntur', 'Kadapa', 
            'Krishna', 'Kurnool', 'Nellore', 'Prakasam', 'Srikakulam', 
            'Visakhapatnam', 'Vizianagaram', 'West Godavari'
        ]
        
        # Years pattern
        self.year_pattern = r'20\d{2}[-–]?\d{0,2}'
    
    def process_extracted_data(self, extracted_data_file: str) -> List[Dict[str, Any]]:
        """Process extracted data into searchable facts"""
        
        try:
            with open(extracted_data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Extracted data file not found: {extracted_data_file}")
            return []
        
        logger.info(f"Processing {len(raw_data)} documents")
        
        facts = []
        fact_id_counter = 1
        
        for doc_name, items in raw_data.items():
            for item in items:
                try:
                    # Process different extraction types
                    if item.get('extraction_method') == 'pymupdf_text':
                        doc_facts = self._process_text_item(item, doc_name, fact_id_counter)
                        facts.extend(doc_facts)
                        fact_id_counter += len(doc_facts)
                    
                    elif item.get('extraction_method') in ['camelot_stream', 'ocr_tesseract']:
                        doc_facts = self._process_table_item(item, doc_name, fact_id_counter)
                        facts.extend(doc_facts)
                        fact_id_counter += len(doc_facts)
                
                except Exception as e:
                    logger.error(f"Error processing item from {doc_name}: {e}")
                    continue
        
        # Save processed facts
        self._save_processed_facts(facts)
        
        logger.info(f"Generated {len(facts)} facts from extracted data")
        return facts
    
    def _process_text_item(self, item: Dict[str, Any], doc_name: str, start_id: int) -> List[Dict[str, Any]]:
        """Extract facts from text content"""
        facts = []
        text = item.get('text', '')
        
        if not text or len(text.strip()) < 50:
            return facts
        
        # Extract year from text
        year = self._extract_year(text)
        
        # Extract district mentions
        districts = self._extract_districts(text)
        
        # Extract education metrics
        for indicator, pattern in self.education_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    value_str = match.group(1).replace(',', '')
                    value = float(value_str)
                    
                    # Get context around the match
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].strip()
                    
                    # Create fact
                    fact = {
                        'fact_id': f"FACT_{start_id + len(facts):06d}",
                        'indicator': indicator.title(),
                        'category': 'total',
                        'district': districts[0] if districts else 'Andhra Pradesh',
                        'year': year,
                        'value': value,
                        'unit': self._infer_unit(indicator, match.group(0)),
                        'source': self._extract_source_type(doc_name),
                        'page_ref': item.get('page', 1),
                        'confidence': 0.8,
                        'span_text': context,
                        'pdf_name': doc_name,
                        'text_id': item.get('text_id', ''),
                        'created_at': datetime.now().isoformat()
                    }
                    
                    facts.append(fact)
                
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse value from match: {e}")
                    continue
        
        return facts
    
    def _process_table_item(self, item: Dict[str, Any], doc_name: str, start_id: int) -> List[Dict[str, Any]]:
        """Extract facts from table data"""
        facts = []
        
        headers = item.get('headers', [])
        rows = item.get('rows', [])
        
        if not headers or not rows:
            return facts
        
        # Try to identify data columns
        numeric_cols = []
        district_col = None
        
        for i, header in enumerate(headers):
            header_lower = header.lower()
            
            # Look for district column
            if any(d.lower() in header_lower for d in self.ap_districts) or 'district' in header_lower:
                district_col = i
            
            # Look for numeric columns
            if any(keyword in header_lower for keyword in ['total', 'count', 'number', 'enrollment', 'schools', 'teachers']):
                numeric_cols.append(i)
        
        # Extract year from table context
        year = self._extract_year(str(item))
        
        # Process each row
        for row_idx, row in enumerate(rows):
            if len(row) != len(headers):
                continue
            
            # Get district
            district = 'Andhra Pradesh'
            if district_col is not None and district_col < len(row):
                district_text = str(row[district_col]).strip()
                for ap_district in self.ap_districts:
                    if ap_district.lower() in district_text.lower():
                        district = ap_district
                        break
            
            # Process numeric values
            for col_idx in numeric_cols:
                if col_idx >= len(row):
                    continue
                
                try:
                    value_text = str(row[col_idx]).strip()
                    value = self._parse_numeric_value(value_text)
                    
                    if value is not None and value > 0:
                        indicator = self._infer_indicator_from_header(headers[col_idx])
                        
                        fact = {
                            'fact_id': f"FACT_{start_id + len(facts):06d}",
                            'indicator': indicator,
                            'category': 'total',
                            'district': district,
                            'year': year,
                            'value': value,
                            'unit': self._infer_unit_from_header(headers[col_idx]),
                            'source': self._extract_source_type(doc_name),
                            'page_ref': item.get('page', 1),
                            'confidence': 0.9,
                            'span_text': f"{headers[col_idx]}: {value_text}",
                            'pdf_name': doc_name,
                            'table_id': item.get('table_id', ''),
                            'created_at': datetime.now().isoformat()
                        }
                        
                        facts.append(fact)
                
                except Exception as e:
                    logger.debug(f"Failed to process table cell [{row_idx}][{col_idx}]: {e}")
                    continue
        
        return facts
    
    def _extract_year(self, text: str) -> str:
        """Extract year from text"""
        year_matches = re.findall(self.year_pattern, text)
        if year_matches:
            # Return the most recent year found
            years = []
            for match in year_matches:
                year_str = match.split('-')[0]  # Take first year if range
                try:
                    year = int(year_str)
                    if 2010 <= year <= 2030:  # Reasonable range
                        years.append(year)
                except ValueError:
                    continue
            if years:
                return str(max(years))
        
        return '2023'  # Default
    
    def _extract_districts(self, text: str) -> List[str]:
        """Extract district mentions from text"""
        districts = []
        for district in self.ap_districts:
            if district.lower() in text.lower():
                districts.append(district)
        return districts
    
    def _extract_source_type(self, doc_name: str) -> str:
        """Extract source type from document name"""
        doc_lower = doc_name.lower()
        if 'cse' in doc_lower:
            return 'CSE'
        elif 'scert' in doc_lower:
            return 'SCERT'
        elif 'go' in doc_lower:
            return 'Government Order'
        elif 'udise' in doc_lower:
            return 'UDISE+'
        else:
            return 'Policy Document'
    
    def _infer_indicator_from_header(self, header: str) -> str:
        """Infer indicator from table header"""
        header_lower = header.lower()
        
        if 'enrol' in header_lower:
            return 'Enrollment'
        elif 'school' in header_lower:
            return 'Schools'
        elif 'teacher' in header_lower:
            return 'Teachers'
        elif 'student' in header_lower:
            return 'Students'
        elif 'class' in header_lower:
            return 'Classrooms'
        elif 'dropout' in header_lower:
            return 'Dropout_Rate'
        elif 'attendance' in header_lower:
            return 'Attendance_Rate'
        else:
            return 'Education_Metric'
    
    def _infer_unit(self, indicator: str, match_text: str) -> str:
        """Infer unit from indicator and context"""
        if '%' in match_text or 'rate' in indicator.lower():
            return 'percentage'
        elif 'ratio' in indicator.lower():
            return 'ratio'
        else:
            return 'count'
    
    def _infer_unit_from_header(self, header: str) -> str:
        """Infer unit from table header"""
        header_lower = header.lower()
        if '%' in header or 'percentage' in header_lower or 'rate' in header_lower:
            return 'percentage'
        elif 'ratio' in header_lower:
            return 'ratio'
        else:
            return 'count'
    
    def _parse_numeric_value(self, value_text: str) -> Optional[float]:
        """Parse numeric value from text"""
        if not value_text or value_text.strip() in ['', '-', 'N/A', 'NA']:
            return None
        
        try:
            # Clean the text
            cleaned = re.sub(r'[^\d.,]', '', value_text)
            cleaned = cleaned.replace(',', '')
            
            if cleaned:
                return float(cleaned)
        except ValueError:
            pass
        
        return None
    
    def _save_processed_facts(self, facts: List[Dict[str, Any]]):
        """Save processed facts to file"""
        try:
            # Save as JSON
            json_file = self.output_dir / "processed_facts.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(facts, f, indent=2, ensure_ascii=False)
            
            # Save as CSV for easier inspection
            if facts:
                df = pd.DataFrame(facts)
                csv_file = self.output_dir / "processed_facts.csv"
                df.to_csv(csv_file, index=False)
            
            # Generate summary
            summary = self._generate_summary(facts)
            summary_file = self.output_dir / "processing_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(facts)} processed facts to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save processed facts: {e}")
    
    def _generate_summary(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate processing summary"""
        if not facts:
            return {'total_facts': 0}
        
        df = pd.DataFrame(facts)
        
        return {
            'total_facts': len(facts),
            'unique_indicators': df['indicator'].nunique(),
            'unique_districts': df['district'].nunique(),
            'unique_years': df['year'].nunique(),
            'sources': df['source'].value_counts().to_dict(),
            'indicators': df['indicator'].value_counts().to_dict(),
            'districts': df['district'].value_counts().to_dict(),
            'years': df['year'].value_counts().to_dict(),
            'processing_timestamp': datetime.now().isoformat()
        }

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process extracted data into searchable facts')
    parser.add_argument('--input-file', default='data/extracted/all_extracted_data.json',
                       help='Input file with extracted data')
    parser.add_argument('--output-dir', default='data/processed',
                       help='Output directory for processed facts')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Process data
    processor = CustomDataProcessor(output_dir=args.output_dir)
    facts = processor.process_extracted_data(args.input_file)
    
    print(f"\n✅ Processing completed: {len(facts)} facts generated")
    print(f"📁 Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()