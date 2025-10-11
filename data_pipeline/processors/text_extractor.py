"""
Text Extraction and Preprocessing for Policy Documents
"""
import fitz  # PyMuPDF
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import pytesseract
from PIL import Image
import numpy as np

# Optional imports
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    doc_id: str
    filename: str
    file_path: str
    document_type: str
    text_length: int
    word_count: int
    page_count: int
    language: str
    processing_date: str
    source_url: Optional[str] = None
    go_number: Optional[str] = None
    circular_number: Optional[str] = None
    document_number: Optional[str] = None

class TextExtractor:
    """Extracts text from various document formats"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR readers
        self.easyocr_reader = None
        self._setup_ocr()
    
    def _setup_ocr(self):
        """Setup OCR engines"""
        if EASYOCR_AVAILABLE:
            try:
                # Initialize EasyOCR for multilingual support
                self.easyocr_reader = easyocr.Reader(['en', 'te'])  # English and Telugu
                logger.info("EasyOCR initialized for English and Telugu")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
        else:
            logger.info("EasyOCR not available, using pytesseract only")
            self.easyocr_reader = None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF using PyMuPDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                text_content.append(text)
            
            doc.close()
            
            # Combine all text
            full_text = "\n".join(text_content)
            
            # Extract metadata
            metadata = self._extract_pdf_metadata(pdf_path, full_text, page_count)
            
            logger.info(f"Extracted text from PDF: {pdf_path} ({len(full_text)} characters)")
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            return "", {}
    
    def extract_text_with_ocr(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF using OCR (for scanned documents)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # Increase resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Use EasyOCR if available
                if self.easyocr_reader:
                    img_array = np.frombuffer(img_data, np.uint8)
                    img = Image.open(io.BytesIO(img_array))
                    img_np = np.array(img)
                    
                    # Extract text using EasyOCR
                    results = self.easyocr_reader.readtext(img_np)
                    page_text = " ".join([result[1] for result in results])
                else:
                    # Fallback to pytesseract
                    img = Image.open(io.BytesIO(img_data))
                    page_text = pytesseract.image_to_string(img)
                
                text_content.append(page_text)
            
            doc.close()
            
            # Combine all text
            full_text = "\n".join(text_content)
            
            # Extract metadata
            metadata = self._extract_pdf_metadata(pdf_path, full_text, page_count)
            metadata['extraction_method'] = 'OCR'
            
            logger.info(f"Extracted text using OCR from PDF: {pdf_path} ({len(full_text)} characters)")
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Failed to extract text with OCR from PDF {pdf_path}: {e}")
            return "", {}
    
    def _extract_pdf_metadata(self, pdf_path: str, text: str, page_count: int) -> Dict[str, Any]:
        """Extract metadata from PDF and text content"""
        filename = Path(pdf_path).name
        
        # Extract document identifiers
        go_number = self._extract_go_number(text)
        circular_number = self._extract_circular_number(text)
        document_number = self._extract_document_number(text)
        
        # Extract dates
        dates = self._extract_dates(text)
        
        # Determine document type
        document_type = self._determine_document_type(filename, text)
        
        # Extract language
        language = self._detect_language(text)
        
        metadata = {
            'filename': filename,
            'file_path': pdf_path,
            'document_type': document_type,
            'text_length': len(text),
            'word_count': len(text.split()),
            'page_count': page_count,
            'language': language,
            'go_number': go_number,
            'circular_number': circular_number,
            'document_number': document_number,
            'dates': dates,
            'extraction_method': 'PyMuPDF'
        }
        
        return metadata
    
    def _extract_go_number(self, text: str) -> Optional[str]:
        """Extract GO number from text"""
        patterns = [
            r'G\.O\.Ms\.No\.\s*(\d+)',
            r'G\.O\.Ms\.\s*(\d+)',
            r'GO\s*(\d+)',
            r'Government\s+Order\s+(\d+)',
            r'G\.O\.\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_circular_number(self, text: str) -> Optional[str]:
        """Extract circular number from text"""
        patterns = [
            r'Circular\s+No\.?\s*(\d+)',
            r'Circ\.\s*(\d+)',
            r'CSE\s*(\d+)',
            r'Notification\s+No\.?\s*(\d+)',
            r'Ref\.\s*No\.?\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_document_number(self, text: str) -> Optional[str]:
        """Extract document number from text"""
        patterns = [
            r'No\.?\s*(\d+)',
            r'Ref\.?\s*(\d+)',
            r'Doc\.?\s*(\d+)',
            r'SCERT\s*(\d+)',
            r'(\d{4})',  # Year pattern
            r'(\d+/\d+)'  # Fraction pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text"""
        date_patterns = [
            r'\d{1,2}-\d{1,2}-\d{4}',  # DD-MM-YYYY
            r'\d{1,2}/\d{1,2}/\d{4}',   # DD/MM/YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}\s+\w+\s+\d{4}',  # DD Month YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        return list(set(dates))  # Remove duplicates
    
    def _determine_document_type(self, filename: str, text: str) -> str:
        """Determine document type from filename and content"""
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        if 'go' in filename_lower or 'government' in filename_lower:
            return 'GO'
        elif 'circular' in filename_lower or 'circular' in text_lower:
            return 'CIRCULAR'
        elif 'scert' in filename_lower or 'scert' in text_lower:
            return 'SCERT'
        elif 'textbook' in filename_lower or 'textbook' in text_lower:
            return 'TEXTBOOK'
        elif 'curriculum' in filename_lower or 'curriculum' in text_lower:
            return 'CURRICULUM'
        elif 'policy' in filename_lower or 'policy' in text_lower:
            return 'POLICY'
        elif 'judgment' in filename_lower or 'court' in text_lower:
            return 'JUDGMENT'
        else:
            return 'DOCUMENT'
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        # Simple language detection based on character patterns
        telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if telugu_chars > english_chars:
            return 'te'  # Telugu
        else:
            return 'en'  # English
    
    def process_document(self, file_path: str, use_ocr: bool = False) -> Dict[str, Any]:
        """
        Process a single document and extract text
        
        Args:
            file_path: Path to document file
            use_ocr: Whether to use OCR for text extraction
            
        Returns:
            Document processing result
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                if use_ocr:
                    text, metadata = self.extract_text_with_ocr(str(file_path))
                else:
                    text, metadata = self.extract_text_from_pdf(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return {}
            
            # Generate document ID
            doc_id = self._generate_doc_id(metadata)
            
            # Create document metadata
            doc_metadata = DocumentMetadata(
                doc_id=doc_id,
                filename=metadata['filename'],
                file_path=str(file_path),
                document_type=metadata['document_type'],
                text_length=metadata['text_length'],
                word_count=metadata['word_count'],
                page_count=metadata['page_count'],
                language=metadata['language'],
                processing_date=str(Path().cwd()),
                source_url=metadata.get('source_url'),
                go_number=metadata.get('go_number'),
                circular_number=metadata.get('circular_number'),
                document_number=metadata.get('document_number')
            )
            
            # Save processed document
            result = self._save_processed_document(doc_metadata, text, metadata)
            
            logger.info(f"Processed document: {doc_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            return {}
    
    def _generate_doc_id(self, metadata: Dict[str, Any]) -> str:
        """Generate unique document ID"""
        doc_type = metadata.get('document_type', 'DOC')
        
        if metadata.get('go_number'):
            return f"GO_{metadata['go_number']}"
        elif metadata.get('circular_number'):
            return f"CIRCULAR_{metadata['circular_number']}"
        elif metadata.get('document_number'):
            return f"{doc_type}_{metadata['document_number']}"
        else:
            # Generate hash-based ID
            filename = metadata.get('filename', 'unknown')
            hash_id = hashlib.md5(filename.encode()).hexdigest()[:8]
            return f"{doc_type}_{hash_id}"
    
    def _save_processed_document(self, metadata: DocumentMetadata, text: str, extra_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Save processed document to file"""
        try:
            # Create output filename
            output_filename = f"{metadata.doc_id}_processed.json"
            output_path = self.output_dir / output_filename
            
            # Prepare document data
            document_data = {
                'metadata': {
                    'doc_id': metadata.doc_id,
                    'filename': metadata.filename,
                    'file_path': metadata.file_path,
                    'document_type': metadata.document_type,
                    'text_length': metadata.text_length,
                    'word_count': metadata.word_count,
                    'page_count': metadata.page_count,
                    'language': metadata.language,
                    'processing_date': metadata.processing_date,
                    'source_url': metadata.source_url,
                    'go_number': metadata.go_number,
                    'circular_number': metadata.circular_number,
                    'document_number': metadata.document_number
                },
                'text': text,
                'extra_metadata': extra_metadata,
                'chunks': []  # Will be populated by chunking process
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved processed document: {output_path}")
            
            return {
                'doc_id': metadata.doc_id,
                'output_path': str(output_path),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Failed to save processed document: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def batch_process_documents(self, input_dir: str, use_ocr: bool = False) -> List[Dict[str, Any]]:
        """
        Batch process all documents in a directory
        
        Args:
            input_dir: Directory containing documents to process
            use_ocr: Whether to use OCR for text extraction
            
        Returns:
            List of processing results
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all PDF files
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return []
        
        logger.info(f"Processing {len(pdf_files)} PDF files")
        
        results = []
        for pdf_file in pdf_files:
            try:
                result = self.process_document(str(pdf_file), use_ocr)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                results.append({
                    'filename': pdf_file.name,
                    'status': 'error',
                    'error': str(e)
                })
        
        logger.info(f"Batch processing completed: {len(results)} documents processed")
        return results

def main():
    """Test the text extractor"""
    extractor = TextExtractor()
    
    # Test with sample documents
    input_dir = "data/raw/gos"
    if Path(input_dir).exists():
        print(f"Processing documents in {input_dir}")
        results = extractor.batch_process_documents(input_dir)
        
        print(f"Processed {len(results)} documents")
        for result in results[:5]:  # Show first 5
            print(f"- {result.get('doc_id', 'unknown')}: {result.get('status', 'unknown')}")
    else:
        print(f"Input directory {input_dir} not found")

if __name__ == "__main__":
    main()
