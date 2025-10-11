"""
Government Orders (GO) Scraper for Andhra Pradesh
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

class GOScraper:
    """Scraper for Government Orders from goir.ap.gov.in"""
    
    def __init__(self, base_url: str = "https://goir.ap.gov.in", output_dir: str = "data/raw/gos"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Keywords to filter education-related GOs
        self.education_keywords = [
            'school', 'education', 'teacher', 'student', 'curriculum', 'examination',
            'board', 'college', 'university', 'academic', 'learning', 'teaching',
            'nep', 'national education policy', 'rte', 'right to education',
            'midday meal', 'scholarship', 'admission', 'enrollment'
        ]
    
    def scrape_gos(self, pages: int = 10, max_documents: int = 100) -> List[Dict[str, Any]]:
        """
        Scrape Government Orders using search functionality
        
        Args:
            pages: Number of search iterations (not used in current implementation)
            max_documents: Maximum number of documents to download
            
        Returns:
            List of scraped document metadata
        """
        logger.info(f"Starting GO scraping using search functionality")
        
        scraped_docs = []
        
        try:
            # Get the main page to extract form data
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract form tokens
            viewstate = soup.find('input', {'name': '__VIEWSTATE'})
            viewstate_gen = soup.find('input', {'name': '__VIEWSTATEGENERATOR'})
            
            if not viewstate or not viewstate_gen:
                logger.error("Could not extract form tokens")
                return scraped_docs
            
            # Search for education-related GOs
            search_terms = ['education', 'school', 'teacher', 'student', 'curriculum', 'NEP']
            
            for search_term in search_terms:
                if len(scraped_docs) >= max_documents:
                    break
                
                logger.info(f"Searching for: {search_term}")
                
                search_data = {
                    '__VIEWSTATE': viewstate.get('value', ''),
                    '__VIEWSTATEGENERATOR': viewstate_gen.get('value', ''),
                    'ctl00$ContentPlaceHolder1$txtGoNo': '',
                    'ctl00$ContentPlaceHolder1$txtfrmdate': '',
                    'ctl00$ContentPlaceHolder1$txttodate': '',
                    'ctl00$ContentPlaceHolder1$fAmount': '',
                    'ctl00$ContentPlaceHolder1$tAmount': '',
                    'ctl00$ContentPlaceHolder1$txtSearchText': search_term,
                    'ctl00$ContentPlaceHolder1$BtnSearch': 'Search'
                }
                
                try:
                    search_response = self.session.post(self.base_url, data=search_data, timeout=30)
                    search_response.raise_for_status()
                    
                    search_soup = BeautifulSoup(search_response.text, 'html.parser')
                    
                    # Look for results in tables
                    tables = search_soup.find_all('table')
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows[1:]:  # Skip header
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                # Look for GO numbers and links
                                for cell in cells:
                                    links = cell.find_all('a', href=True)
                                    for link in links:
                                        link_text = link.get_text(strip=True)
                                        link_href = link.get('href')
                                        
                                        if self._is_education_related(link_text):
                                            doc_info = self._process_go_link(link, link_href)
                                            if doc_info and len(scraped_docs) < max_documents:
                                                scraped_docs.append(doc_info)
                    
                    # Rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Search failed for '{search_term}': {e}")
                    continue
            
            # If search didn't work, try to scrape from Reports page
            if not scraped_docs:
                scraped_docs = self._scrape_from_reports(max_documents)
            
        except Exception as e:
            logger.error(f"GO scraping failed: {e}")
        
        logger.info(f"Scraped {len(scraped_docs)} education-related GOs")
        return scraped_docs
    
    def _scrape_from_reports(self, max_documents: int) -> List[Dict[str, Any]]:
        """Fallback method to scrape from Reports page"""
        scraped_docs = []
        
        try:
            response = self.session.get(f"{self.base_url}/Reports.aspx", timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for any downloadable documents
            links = soup.find_all('a', href=True)
            
            for link in links:
                if len(scraped_docs) >= max_documents:
                    break
                
                link_text = link.get_text(strip=True)
                link_href = link.get('href')
                
                # Check if it's a document link
                if link_href and ('.pdf' in link_href.lower() or self._is_education_related(link_text)):
                    doc_info = self._process_go_link(link, link_href)
                    if doc_info:
                        scraped_docs.append(doc_info)
            
        except Exception as e:
            logger.error(f"Failed to scrape from Reports page: {e}")
        
        return scraped_docs
    
    def _is_education_related(self, text: str) -> bool:
        """Check if text contains education-related keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.education_keywords)
    
    def _process_go_link(self, link_element, link_href: str) -> Optional[Dict[str, Any]]:
        """
        Process a GO link and download if it's a PDF
        
        Args:
            link_element: BeautifulSoup link element
            link_href: Link href attribute
            
        Returns:
            Document metadata if successful, None otherwise
        """
        try:
            # Construct full URL
            full_url = urljoin(self.base_url, link_href)
            
            # Check if it's a PDF link
            if not full_url.lower().endswith('.pdf'):
                return None
            
            # Extract GO number and title
            link_text = link_element.get_text(strip=True)
            go_number = self._extract_go_number(link_text)
            
            # Download the PDF
            pdf_filename = self._download_pdf(full_url, go_number)
            if not pdf_filename:
                return None
            
            # Extract metadata
            metadata = {
                'document_id': f"GO_{go_number or 'unknown'}",
                'title': link_text,
                'go_number': go_number,
                'source_url': full_url,
                'filename': pdf_filename,
                'file_path': str(self.output_dir / pdf_filename),
                'document_type': 'GO',
                'scraped_at': time.time(),
                'status': 'downloaded'
            }
            
            logger.info(f"Downloaded GO: {go_number}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to process GO link: {e}")
            return None
    
    def _extract_go_number(self, text: str) -> Optional[str]:
        """Extract GO number from text"""
        # Common GO number patterns
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
    
    def _download_pdf(self, url: str, go_number: Optional[str]) -> Optional[str]:
        """
        Download PDF from URL
        
        Args:
            url: PDF URL
            go_number: GO number for filename
            
        Returns:
            Filename if successful, None otherwise
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Generate filename
            if go_number:
                filename = f"GO_{go_number}.pdf"
            else:
                # Extract filename from URL
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                if not filename.endswith('.pdf'):
                    filename = f"GO_{int(time.time())}.pdf"
            
            # Save PDF
            file_path = self.output_dir / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.debug(f"Downloaded PDF: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to download PDF from {url}: {e}")
            return None
    
    def scrape_specific_go(self, go_number: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a specific GO by number
        
        Args:
            go_number: GO number to search for
            
        Returns:
            Document metadata if found
        """
        logger.info(f"Searching for specific GO: {go_number}")
        
        try:
            # Search for the GO
            search_url = f"{self.base_url}/Search.aspx"
            search_params = {
                'search': go_number,
                'type': 'orders'
            }
            
            response = self.session.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find matching links
            links = soup.find_all('a', href=True)
            
            for link in links:
                link_text = link.get_text(strip=True)
                if go_number in link_text:
                    link_href = link.get('href')
                    return self._process_go_link(link, link_href)
            
            logger.warning(f"GO {go_number} not found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to search for GO {go_number}: {e}")
            return None
    
    def get_recent_gos(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get recent GOs from the last N days
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent GO metadata
        """
        logger.info(f"Getting GOs from last {days} days")
        
        recent_docs = []
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        # This would need to be implemented based on the actual website structure
        # For now, we'll scrape recent pages
        pages_to_check = min(5, days // 7 + 1)  # Estimate pages to check
        
        for page in range(1, pages_to_check + 1):
            try:
                url = f"{self.base_url}/Orders.aspx?page={page}"
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for date information in the page
                # This would need to be customized based on actual website structure
                links = soup.find_all('a', href=True)
                
                for link in links:
                    link_text = link.get_text(strip=True).lower()
                    if self._is_education_related(link_text):
                        doc_info = self._process_go_link(link, link.get('href'))
                        if doc_info:
                            # Check if document is recent enough
                            if doc_info.get('scraped_at', 0) >= cutoff_time:
                                recent_docs.append(doc_info)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Failed to get recent GOs from page {page}: {e}")
                continue
        
        logger.info(f"Found {len(recent_docs)} recent education GOs")
        return recent_docs
    
    def save_metadata(self, documents: List[Dict[str, Any]], filename: str = "go_metadata.json"):
        """Save scraped document metadata to JSON file"""
        try:
            metadata_file = self.output_dir / filename
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved metadata for {len(documents)} documents to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def load_metadata(self, filename: str = "go_metadata.json") -> List[Dict[str, Any]]:
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
    """Test the GO scraper"""
    scraper = GOScraper()
    
    # Scrape recent GOs
    print("Scraping recent Government Orders...")
    documents = scraper.scrape_gos(pages=5, max_documents=20)
    
    # Save metadata
    scraper.save_metadata(documents)
    
    print(f"Scraped {len(documents)} documents")
    for doc in documents[:5]:  # Show first 5
        print(f"- {doc['title']} ({doc['go_number']})")

if __name__ == "__main__":
    main()

