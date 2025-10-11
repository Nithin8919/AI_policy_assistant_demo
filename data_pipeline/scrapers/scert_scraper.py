"""
SCERT Scraper for Andhra Pradesh State Council of Educational Research and Training
"""
import requests
import os
import time
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re
from urllib.parse import urljoin, urlparse
import json

logger = logging.getLogger(__name__)

class SCERTScraper:
    """Scraper for SCERT Andhra Pradesh (apscert.gov.in)"""
    
    def __init__(self, base_url: str = "https://cse.ap.gov.in", output_dir: str = "data/raw/scert"):
        # Using CSE portal as alternative source since SCERT site is not accessible
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Keywords for curriculum and training materials
        self.curriculum_keywords = [
            'curriculum', 'syllabus', 'textbook', 'training', 'teacher', 'education',
            'learning', 'teaching', 'pedagogy', 'assessment', 'evaluation', 'examination',
            'material', 'resource', 'guide', 'manual', 'handbook', 'framework',
            'policy', 'guidelines', 'standards', 'competencies', 'skills'
        ]
    
    def scrape_curriculum_materials(self, max_documents: int = 100) -> List[Dict[str, Any]]:
        """
        Scrape curriculum materials and resources from CSE portal (SCERT alternative)
        
        Args:
            max_documents: Maximum number of documents to download
            
        Returns:
            List of scraped document metadata
        """
        logger.info("Starting curriculum materials scraping from CSE portal")
        
        scraped_docs = []
        
        try:
            # Scrape from CSE portal home page for curriculum-related documents
            response = self.session.get(f"{self.base_url}/home", timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            
            for link in links:
                if len(scraped_docs) >= max_documents:
                    break
                
                link_href = link.get('href')
                link_text = link.get_text(strip=True)
                
                # Look for curriculum-related documents
                if link_href and ('.pdf' in link_href.lower() or self._is_curriculum_related(link_text)):
                    doc_info = self._process_scert_link(link_text, link_href, 'curriculum')
                    if doc_info:
                        scraped_docs.append(doc_info)
            
            # Also try to scrape from MIS download section
            if len(scraped_docs) < max_documents:
                additional_docs = self._scrape_mis_documents(max_documents - len(scraped_docs))
                scraped_docs.extend(additional_docs)
            
        except Exception as e:
            logger.error(f"Curriculum materials scraping failed: {e}")
        
        logger.info(f"Scraped {len(scraped_docs)} curriculum materials")
        return scraped_docs
    
    def _scrape_mis_documents(self, max_documents: int) -> List[Dict[str, Any]]:
        """Scrape MIS documents from CSE portal"""
        scraped_docs = []
        
        try:
            # Look for MIS download links
            response = self.session.get(f"{self.base_url}/home", timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            
            for link in links:
                if len(scraped_docs) >= max_documents:
                    break
                
                link_href = link.get('href')
                link_text = link.get_text(strip=True)
                
                # Look for MIS or statistics related documents
                if link_href and ('mis' in link_href.lower() or 'statistics' in link_text.lower() or 'educational' in link_text.lower()):
                    doc_info = self._process_scert_link(link_text, link_href, 'mis')
                    if doc_info:
                        scraped_docs.append(doc_info)
            
        except Exception as e:
            logger.error(f"Failed to scrape MIS documents: {e}")
        
        return scraped_docs
    
    def _scrape_section(self, url: str, section_name: str, max_docs: int) -> List[Dict[str, Any]]:
        """
        Scrape a specific section of SCERT website
        
        Args:
            url: Section URL
            section_name: Name of the section
            max_docs: Maximum documents to scrape from this section
            
        Returns:
            List of document metadata
        """
        docs = []
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links
            links = soup.find_all('a', href=True)
            
            for link in links:
                if len(docs) >= max_docs:
                    break
                
                link_text = link.get_text(strip=True).lower()
                link_href = link.get('href')
                
                if self._is_curriculum_related(link_text) and link_href:
                    doc_info = self._process_scert_link(link_text, link_href, section_name)
                    if doc_info:
                        docs.append(doc_info)
            
        except Exception as e:
            logger.debug(f"Failed to scrape section {section_name}: {e}")
        
        return docs
    
    def _is_curriculum_related(self, text: str) -> bool:
        """Check if text contains curriculum-related keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.curriculum_keywords)
    
    def _process_scert_link(self, link_text: str, link_href: str, section: str) -> Optional[Dict[str, Any]]:
        """
        Process a SCERT link and download if it's a document
        
        Args:
            link_text: Link text
            link_href: Link href
            section: Section name
            
        Returns:
            Document metadata if successful, None otherwise
        """
        try:
            # Construct full URL
            full_url = urljoin(self.base_url, link_href)
            
            # Check if it's a document link
            if not self._is_document_link(full_url):
                return None
            
            # Extract document information
            doc_type = self._extract_document_type(link_text)
            doc_number = self._extract_document_number(link_text)
            
            # Download the document
            filename = self._download_document(full_url, doc_number, doc_type)
            if not filename:
                return None
            
            # Extract metadata
            metadata = {
                'document_id': f"SCERT_{doc_number or 'unknown'}",
                'title': link_text,
                'document_number': doc_number,
                'document_type': doc_type,
                'section': section,
                'source_url': full_url,
                'filename': filename,
                'file_path': str(self.output_dir / filename),
                'scraped_at': time.time(),
                'status': 'downloaded'
            }
            
            logger.info(f"Downloaded SCERT document: {doc_type} - {doc_number}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to process SCERT link: {e}")
            return None
    
    def _is_document_link(self, url: str) -> bool:
        """Check if URL points to a document"""
        document_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        url_lower = url.lower()
        
        return any(url_lower.endswith(ext) for ext in document_extensions)
    
    def _extract_document_type(self, text: str) -> str:
        """Extract document type from text"""
        text_lower = text.lower()
        
        if 'textbook' in text_lower or 'book' in text_lower:
            return 'TEXTBOOK'
        elif 'curriculum' in text_lower or 'syllabus' in text_lower:
            return 'CURRICULUM'
        elif 'training' in text_lower or 'manual' in text_lower:
            return 'TRAINING_MATERIAL'
        elif 'assessment' in text_lower or 'evaluation' in text_lower:
            return 'ASSESSMENT'
        elif 'policy' in text_lower or 'guideline' in text_lower:
            return 'POLICY'
        elif 'research' in text_lower or 'paper' in text_lower:
            return 'RESEARCH'
        else:
            return 'DOCUMENT'
    
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
    
    def _download_document(self, url: str, doc_number: Optional[str], doc_type: str) -> Optional[str]:
        """
        Download document from URL
        
        Args:
            url: Document URL
            doc_number: Document number for filename
            doc_type: Document type
            
        Returns:
            Filename if successful, None otherwise
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file extension
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type:
                ext = '.pdf'
            elif 'word' in content_type or 'document' in content_type:
                ext = '.docx'
            elif 'excel' in content_type or 'spreadsheet' in content_type:
                ext = '.xlsx'
            elif 'powerpoint' in content_type or 'presentation' in content_type:
                ext = '.pptx'
            else:
                # Try to get extension from URL
                parsed_url = urlparse(url)
                ext = os.path.splitext(parsed_url.path)[1]
                if not ext:
                    ext = '.pdf'  # Default to PDF
            
            # Generate filename
            if doc_number:
                filename = f"SCERT_{doc_type}_{doc_number}{ext}"
            else:
                filename = f"SCERT_{doc_type}_{int(time.time())}{ext}"
            
            # Save document
            file_path = self.output_dir / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.debug(f"Downloaded document: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to download document from {url}: {e}")
            return None
    
    def scrape_textbooks(self, classes: List[str] = None) -> List[Dict[str, Any]]:
        """
        Scrape textbooks for specific classes
        
        Args:
            classes: List of class names (e.g., ['1', '2', '3', '4', '5'])
            
        Returns:
            List of textbook metadata
        """
        if classes is None:
            classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        logger.info(f"Scraping textbooks for classes: {classes}")
        
        textbooks = []
        
        try:
            for class_name in classes:
                # Common textbook URLs
                textbook_urls = [
                    f"{self.base_url}/textbooks/class-{class_name}",
                    f"{self.base_url}/books/class-{class_name}",
                    f"{self.base_url}/materials/class-{class_name}"
                ]
                
                for url in textbook_urls:
                    try:
                        response = self.session.get(url, timeout=30)
                        response.raise_for_status()
                        
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find textbook links
                        links = soup.find_all('a', href=True)
                        
                        for link in links:
                            link_text = link.get_text(strip=True).lower()
                            link_href = link.get('href')
                            
                            if 'textbook' in link_text and link_href:
                                doc_info = self._process_scert_link(
                                    link_text, link_href, f"class_{class_name}"
                                )
                                if doc_info:
                                    doc_info['class'] = class_name
                                    textbooks.append(doc_info)
                        
                        time.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        logger.debug(f"Failed to scrape textbooks for class {class_name}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Textbook scraping failed: {e}")
        
        logger.info(f"Scraped {len(textbooks)} textbooks")
        return textbooks
    
    def scrape_training_materials(self) -> List[Dict[str, Any]]:
        """
        Scrape teacher training materials
        
        Returns:
            List of training material metadata
        """
        logger.info("Scraping teacher training materials")
        
        training_materials = []
        
        try:
            # Common training material URLs
            training_urls = [
                f"{self.base_url}/training",
                f"{self.base_url}/teacher-training",
                f"{self.base_url}/professional-development",
                f"{self.base_url}/workshops"
            ]
            
            for url in training_urls:
                try:
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find training material links
                    links = soup.find_all('a', href=True)
                    
                    for link in links:
                        link_text = link.get_text(strip=True).lower()
                        link_href = link.get('href')
                        
                        if any(keyword in link_text for keyword in ['training', 'manual', 'guide', 'handbook']):
                            doc_info = self._process_scert_link(link_text, link_href, 'training')
                            if doc_info:
                                training_materials.append(doc_info)
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Failed to scrape training materials from {url}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Training materials scraping failed: {e}")
        
        logger.info(f"Scraped {len(training_materials)} training materials")
        return training_materials
    
    def save_metadata(self, documents: List[Dict[str, Any]], filename: str = "scert_metadata.json"):
        """Save scraped document metadata to JSON file"""
        try:
            metadata_file = self.output_dir / filename
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved metadata for {len(documents)} documents to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def load_metadata(self, filename: str = "scert_metadata.json") -> List[Dict[str, Any]]:
        """Load scraped document metadata from JSON file"""
        try:
            metadata_file = self.output_dir / filename
            if not metadata_file.exists():
                return []
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            logger.info(f"Loaded metadata for {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return []

def main():
    """Test the SCERT scraper"""
    scraper = SCERTScraper()
    
    # Scrape curriculum materials
    print("Scraping SCERT curriculum materials...")
    documents = scraper.scrape_curriculum_materials(max_documents=30)
    
    # Scrape textbooks
    print("Scraping textbooks...")
    textbooks = scraper.scrape_textbooks(classes=['1', '2', '3'])
    
    # Scrape training materials
    print("Scraping training materials...")
    training_materials = scraper.scrape_training_materials()
    
    # Combine all documents
    all_documents = documents + textbooks + training_materials
    
    # Save metadata
    scraper.save_metadata(all_documents)
    
    print(f"Scraped {len(all_documents)} total documents")
    for doc in all_documents[:5]:  # Show first 5
        print(f"- {doc['title']} ({doc['document_type']})")

if __name__ == "__main__":
    main()
