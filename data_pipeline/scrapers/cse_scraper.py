"""
CSE Portal Scraper for Andhra Pradesh School Education
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
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger(__name__)

class CSEPortalScraper:
    """Scraper for CSE Portal (cse.ap.gov.in)"""
    
    def __init__(self, base_url: str = "https://cse.ap.gov.in", output_dir: str = "data/raw/cse"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Keywords for education-related content
        self.education_keywords = [
            'school', 'education', 'teacher', 'student', 'curriculum', 'examination',
            'board', 'academic', 'learning', 'teaching', 'admission', 'enrollment',
            'policy', 'guidelines', 'circular', 'notification', 'order'
        ]
        
        # Selenium driver for JavaScript-heavy pages
        self.driver = None
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            
            logger.info("Selenium WebDriver initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup Selenium: {e}")
            self.driver = None
    
    def scrape_circulars(self, max_documents: int = 50) -> List[Dict[str, Any]]:
        """
        Scrape circulars and notifications from CSE portal
        
        Args:
            max_documents: Maximum number of documents to download
            
        Returns:
            List of scraped document metadata
        """
        logger.info("Starting CSE portal circular scraping")
        
        scraped_docs = []
        
        try:
            # First try the home page which has PDF links
            scraped_docs = self._scrape_from_home_page(max_documents)
            
            # If not enough documents, try other pages
            if len(scraped_docs) < max_documents:
                additional_docs = self._scrape_from_other_pages(max_documents - len(scraped_docs))
                scraped_docs.extend(additional_docs)
            
        except Exception as e:
            logger.error(f"CSE portal scraping failed: {e}")
        
        logger.info(f"Scraped {len(scraped_docs)} documents from CSE portal")
        return scraped_docs
    
    def _scrape_from_home_page(self, max_documents: int) -> List[Dict[str, Any]]:
        """Scrape PDFs from CSE home page"""
        scraped_docs = []
        
        try:
            response = self.session.get(f"{self.base_url}/home", timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            
            for link in links:
                if len(scraped_docs) >= max_documents:
                    break
                
                link_href = link.get('href')
                link_text = link.get_text(strip=True)
                
                # Check if it's a PDF link
                if link_href and '.pdf' in link_href.lower():
                    doc_info = self._process_circular_link(link_text, link_href)
                    if doc_info:
                        scraped_docs.append(doc_info)
            
        except Exception as e:
            logger.error(f"Failed to scrape from home page: {e}")
        
        return scraped_docs
    
    def _scrape_from_other_pages(self, max_documents: int) -> List[Dict[str, Any]]:
        """Scrape from other CSE pages"""
        scraped_docs = []
        
        try:
            # Try RTE notifications page
            response = self.session.get(f"{self.base_url}/RteNotificationsPage", timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            
            for link in links:
                if len(scraped_docs) >= max_documents:
                    break
                
                link_href = link.get('href')
                link_text = link.get_text(strip=True)
                
                if link_href and ('.pdf' in link_href.lower() or self._is_education_related(link_text)):
                    doc_info = self._process_circular_link(link_text, link_href)
                    if doc_info:
                        scraped_docs.append(doc_info)
            
        except Exception as e:
            logger.error(f"Failed to scrape from other pages: {e}")
        
        return scraped_docs
    
    def _scrape_with_selenium(self, max_documents: int) -> List[Dict[str, Any]]:
        """Scrape using Selenium WebDriver"""
        scraped_docs = []
        
        try:
            # Navigate to circulars page
            circulars_url = f"{self.base_url}/circulars"
            self.driver.get(circulars_url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Find all circular links
            links = self.driver.find_elements(By.TAG_NAME, "a")
            
            for link in links:
                if len(scraped_docs) >= max_documents:
                    break
                
                try:
                    link_text = link.text.strip().lower()
                    link_href = link.get_attribute('href')
                    
                    if self._is_education_related(link_text) and link_href:
                        doc_info = self._process_circular_link(link_text, link_href)
                        if doc_info:
                            scraped_docs.append(doc_info)
                    
                except Exception as e:
                    logger.debug(f"Failed to process link: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Selenium scraping failed: {e}")
        
        return scraped_docs
    
    def _scrape_with_requests(self, max_documents: int) -> List[Dict[str, Any]]:
        """Scrape using requests (fallback)"""
        scraped_docs = []
        
        try:
            # Try different possible URLs
            urls_to_try = [
                f"{self.base_url}/circulars",
                f"{self.base_url}/notifications",
                f"{self.base_url}/announcements",
                f"{self.base_url}/"
            ]
            
            for url in urls_to_try:
                if len(scraped_docs) >= max_documents:
                    break
                
                try:
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    links = soup.find_all('a', href=True)
                    
                    for link in links:
                        if len(scraped_docs) >= max_documents:
                            break
                        
                        link_text = link.get_text(strip=True).lower()
                        link_href = link.get('href')
                        
                        if self._is_education_related(link_text) and link_href:
                            doc_info = self._process_circular_link(link_text, link_href)
                            if doc_info:
                                scraped_docs.append(doc_info)
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Failed to scrape {url}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Requests scraping failed: {e}")
        
        return scraped_docs
    
    def _is_education_related(self, text: str) -> bool:
        """Check if text contains education-related keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.education_keywords)
    
    def _process_circular_link(self, link_text: str, link_href: str) -> Optional[Dict[str, Any]]:
        """
        Process a circular link and download if it's a document
        
        Args:
            link_text: Link text
            link_href: Link href
            
        Returns:
            Document metadata if successful, None otherwise
        """
        try:
            # Construct full URL
            full_url = urljoin(self.base_url, link_href)
            
            # Check if it's a document link
            if not self._is_document_link(full_url):
                return None
            
            # Extract circular number and title
            circular_number = self._extract_circular_number(link_text)
            
            # Download the document
            filename = self._download_document(full_url, circular_number)
            if not filename:
                return None
            
            # Extract metadata
            metadata = {
                'document_id': f"CSE_{circular_number or 'unknown'}",
                'title': link_text,
                'circular_number': circular_number,
                'source_url': full_url,
                'filename': filename,
                'file_path': str(self.output_dir / filename),
                'document_type': 'CSE_CIRCULAR',
                'scraped_at': time.time(),
                'status': 'downloaded'
            }
            
            logger.info(f"Downloaded CSE circular: {circular_number}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to process circular link: {e}")
            return None
    
    def _is_document_link(self, url: str) -> bool:
        """Check if URL points to a document"""
        document_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        url_lower = url.lower()
        
        return any(url_lower.endswith(ext) for ext in document_extensions)
    
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
    
    def _download_document(self, url: str, circular_number: Optional[str]) -> Optional[str]:
        """
        Download document from URL
        
        Args:
            url: Document URL
            circular_number: Circular number for filename
            
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
            if circular_number:
                filename = f"CSE_{circular_number}{ext}"
            else:
                filename = f"CSE_{int(time.time())}{ext}"
            
            # Save document
            file_path = self.output_dir / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.debug(f"Downloaded document: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to download document from {url}: {e}")
            return None
    
    def scrape_school_data(self) -> List[Dict[str, Any]]:
        """
        Scrape school data and statistics
        
        Returns:
            List of school data
        """
        logger.info("Scraping school data from CSE portal")
        
        school_data = []
        
        try:
            # This would need to be customized based on actual portal structure
            # Common endpoints for school data
            data_urls = [
                f"{self.base_url}/schools",
                f"{self.base_url}/statistics",
                f"{self.base_url}/reports"
            ]
            
            for url in data_urls:
                try:
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for data tables or download links
                    tables = soup.find_all('table')
                    for table in tables:
                        data = self._extract_table_data(table)
                        school_data.extend(data)
                    
                    # Look for CSV/Excel download links
                    download_links = soup.find_all('a', href=True)
                    for link in download_links:
                        href = link.get('href')
                        if href and any(ext in href.lower() for ext in ['.csv', '.xlsx', '.xls']):
                            self._download_data_file(href)
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Failed to scrape data from {url}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"School data scraping failed: {e}")
        
        logger.info(f"Scraped {len(school_data)} school data records")
        return school_data
    
    def _extract_table_data(self, table) -> List[Dict[str, Any]]:
        """Extract data from HTML table"""
        data = []
        
        try:
            rows = table.find_all('tr')
            if not rows:
                return data
            
            # Get headers
            headers = []
            header_row = rows[0]
            for th in header_row.find_all(['th', 'td']):
                headers.append(th.get_text(strip=True))
            
            # Get data rows
            for row in rows[1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) == len(headers):
                    row_data = {}
                    for i, cell in enumerate(cells):
                        row_data[headers[i]] = cell.get_text(strip=True)
                    data.append(row_data)
            
        except Exception as e:
            logger.debug(f"Failed to extract table data: {e}")
        
        return data
    
    def _download_data_file(self, url: str):
        """Download data file (CSV, Excel, etc.)"""
        try:
            full_url = urljoin(self.base_url, url)
            response = self.session.get(full_url, timeout=30)
            response.raise_for_status()
            
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = f"data_{int(time.time())}.csv"
            
            file_path = self.output_dir / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded data file: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to download data file {url}: {e}")
    
    def save_metadata(self, documents: List[Dict[str, Any]], filename: str = "cse_metadata.json"):
        """Save scraped document metadata to JSON file"""
        try:
            metadata_file = self.output_dir / filename
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved metadata for {len(documents)} documents to {metadata_file}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def load_metadata(self, filename: str = "cse_metadata.json") -> List[Dict[str, Any]]:
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
    """Test the CSE portal scraper"""
    scraper = CSEPortalScraper()
    
    # Scrape circulars
    print("Scraping CSE portal circulars...")
    documents = scraper.scrape_circulars(max_documents=20)
    
    # Scrape school data
    print("Scraping school data...")
    school_data = scraper.scrape_school_data()
    
    # Save metadata
    scraper.save_metadata(documents)
    
    print(f"Scraped {len(documents)} circulars and {len(school_data)} school data records")
    for doc in documents[:5]:  # Show first 5
        print(f"- {doc['title']} ({doc['circular_number']})")

if __name__ == "__main__":
    main()
