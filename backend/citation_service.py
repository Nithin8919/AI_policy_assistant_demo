#!/usr/bin/env python3
"""
Citation Service - Advanced Citation and Source Management
Handles citation generation, source tracking, and reference management
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import json
import hashlib
from datetime import datetime, date
from pathlib import Path
import re
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class CitationStyle(Enum):
    """Supported citation styles"""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    VANCOUVER = "vancouver"
    HARVARD = "harvard"

class SourceType(Enum):
    """Types of sources"""
    GOVERNMENT_DOCUMENT = "government_document"
    POLICY_DOCUMENT = "policy_document"
    STATISTICAL_REPORT = "statistical_report"
    RESEARCH_PAPER = "research_paper"
    CIRCULAR = "circular"
    NOTIFICATION = "notification"
    WEB_RESOURCE = "web_resource"
    DATABASE = "database"

@dataclass
class SourceMetadata:
    """Metadata for a source document"""
    source_id: str
    title: str
    authors: List[str]
    organization: str
    publication_date: Optional[date]
    access_date: date
    source_type: SourceType
    url: Optional[str] = None
    doi: Optional[str] = None
    page_numbers: Optional[Tuple[int, int]] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    edition: Optional[str] = None
    location: Optional[str] = None
    publisher: Optional[str] = None
    isbn: Optional[str] = None
    language: str = "English"
    
    def __post_init__(self):
        if isinstance(self.source_type, str):
            self.source_type = SourceType(self.source_type)

@dataclass
class Citation:
    """A formatted citation"""
    citation_id: str
    source_metadata: SourceMetadata
    style: CitationStyle
    formatted_citation: str
    in_text_citation: str
    created_at: datetime
    page_reference: Optional[str] = None
    quote_text: Optional[str] = None
    context: Optional[str] = None

class CitationService:
    """Service for managing citations and source references"""
    
    def __init__(self, citation_database_path: Optional[str] = None):
        self.citation_database_path = citation_database_path or "data/citations.json"
        self.citations_cache: Dict[str, Citation] = {}
        self.sources_cache: Dict[str, SourceMetadata] = {}
        
        # Citation patterns for different styles
        self.style_formatters = {
            CitationStyle.APA: self._format_apa,
            CitationStyle.MLA: self._format_mla,
            CitationStyle.CHICAGO: self._format_chicago,
            CitationStyle.IEEE: self._format_ieee,
            CitationStyle.VANCOUVER: self._format_vancouver,
            CitationStyle.HARVARD: self._format_harvard
        }
        
        # Load existing citations
        self._load_citations()
        
        logger.info(f"Citation Service initialized with {len(self.citations_cache)} cached citations")
    
    def add_source(self, metadata: SourceMetadata) -> str:
        """Add a new source to the database"""
        try:
            source_id = metadata.source_id
            self.sources_cache[source_id] = metadata
            
            # Save to disk
            self._save_citations()
            
            logger.info(f"Added source: {source_id}")
            return source_id
            
        except Exception as e:
            logger.error(f"Failed to add source: {e}")
            raise
    
    def create_citation(
        self,
        source_id: str,
        style: CitationStyle = CitationStyle.APA,
        page_reference: Optional[str] = None,
        quote_text: Optional[str] = None,
        context: Optional[str] = None
    ) -> Citation:
        """Create a formatted citation for a source"""
        try:
            if source_id not in self.sources_cache:
                raise ValueError(f"Source {source_id} not found in database")
            
            source_metadata = self.sources_cache[source_id]
            
            # Generate citation ID
            citation_id = self._generate_citation_id(source_id, style, page_reference)
            
            # Check if citation already exists
            if citation_id in self.citations_cache:
                return self.citations_cache[citation_id]
            
            # Format citation based on style
            formatter = self.style_formatters.get(style, self._format_apa)
            formatted_citation, in_text_citation = formatter(source_metadata, page_reference)
            
            # Create citation object
            citation = Citation(
                citation_id=citation_id,
                source_metadata=source_metadata,
                style=style,
                formatted_citation=formatted_citation,
                in_text_citation=in_text_citation,
                created_at=datetime.now(),
                page_reference=page_reference,
                quote_text=quote_text,
                context=context
            )
            
            # Cache and save
            self.citations_cache[citation_id] = citation
            self._save_citations()
            
            logger.info(f"Created citation: {citation_id}")
            return citation
            
        except Exception as e:
            logger.error(f"Failed to create citation for {source_id}: {e}")
            raise
    
    def get_citations_for_search_results(
        self,
        search_results: List[Dict[str, Any]],
        style: CitationStyle = CitationStyle.APA
    ) -> List[Dict[str, Any]]:
        """Generate citations for search results"""
        try:
            enhanced_results = []
            
            for result in search_results:
                enhanced_result = result.copy()
                
                # Try to create citation from result metadata
                citation = self._create_citation_from_result(result, style)
                if citation:
                    enhanced_result['citation'] = {
                        'formatted': citation.formatted_citation,
                        'in_text': citation.in_text_citation,
                        'style': citation.style.value,
                        'citation_id': citation.citation_id
                    }
                
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to generate citations for search results: {e}")
            return search_results  # Return original results if citation fails
    
    def _create_citation_from_result(
        self,
        result: Dict[str, Any],
        style: CitationStyle
    ) -> Optional[Citation]:
        """Create citation from search result metadata"""
        try:
            # Extract metadata from result
            source_document = result.get('source_document') or result.get('source', '')
            page_ref = result.get('page_ref') or result.get('page_number')
            
            if not source_document:
                return None
            
            # Generate source ID from document name
            source_id = self._generate_source_id(source_document)
            
            # Check if source already exists
            if source_id not in self.sources_cache:
                # Create new source metadata from result
                source_metadata = self._extract_source_metadata_from_result(result, source_id)
                self.add_source(source_metadata)
            
            # Create citation
            return self.create_citation(
                source_id=source_id,
                style=style,
                page_reference=str(page_ref) if page_ref else None,
                context=result.get('content') or result.get('span_text')
            )
            
        except Exception as e:
            logger.error(f"Failed to create citation from result: {e}")
            return None
    
    def _extract_source_metadata_from_result(
        self,
        result: Dict[str, Any],
        source_id: str
    ) -> SourceMetadata:
        """Extract source metadata from search result"""
        
        source_document = result.get('source_document') or result.get('source', '')
        year = result.get('year')
        
        # Determine source type from document name
        source_type = self._determine_source_type(source_document)
        
        # Extract organization from document name or use default
        organization = self._extract_organization(source_document)
        
        # Create metadata
        return SourceMetadata(
            source_id=source_id,
            title=self._clean_document_title(source_document),
            authors=[],  # Usually not available in search results
            organization=organization,
            publication_date=date(year, 1, 1) if year and isinstance(year, int) else None,
            access_date=date.today(),
            source_type=source_type,
            language="English"
        )
    
    def _generate_source_id(self, source_document: str) -> str:
        """Generate unique source ID from document name"""
        # Clean and normalize document name
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', source_document.lower())
        clean_name = re.sub(r'_+', '_', clean_name)
        return f"source_{clean_name}"[:50]  # Limit length
    
    def _generate_citation_id(
        self,
        source_id: str,
        style: CitationStyle,
        page_reference: Optional[str] = None
    ) -> str:
        """Generate unique citation ID"""
        components = [source_id, style.value]
        if page_reference:
            components.append(str(page_reference))
        
        combined = "_".join(components)
        hash_obj = hashlib.md5(combined.encode())
        return f"cite_{hash_obj.hexdigest()[:8]}"
    
    def _determine_source_type(self, source_document: str) -> SourceType:
        """Determine source type from document name"""
        source_lower = source_document.lower()
        
        if any(word in source_lower for word in ['go', 'government_order', 'order']):
            return SourceType.GOVERNMENT_DOCUMENT
        elif any(word in source_lower for word in ['circular', 'memo']):
            return SourceType.CIRCULAR
        elif any(word in source_lower for word in ['notification', 'notice']):
            return SourceType.NOTIFICATION
        elif any(word in source_lower for word in ['policy', 'guideline']):
            return SourceType.POLICY_DOCUMENT
        elif any(word in source_lower for word in ['report', 'statistics', 'data']):
            return SourceType.STATISTICAL_REPORT
        else:
            return SourceType.GOVERNMENT_DOCUMENT
    
    def _extract_organization(self, source_document: str) -> str:
        """Extract organization from document name"""
        source_lower = source_document.lower()
        
        if 'ap' in source_lower or 'andhra pradesh' in source_lower:
            return "Government of Andhra Pradesh"
        elif 'cse' in source_lower:
            return "Commissioner of School Education, AP"
        elif 'scert' in source_lower:
            return "State Council of Educational Research and Training, AP"
        elif 'udise' in source_lower:
            return "UDISE+ Database"
        else:
            return "Government of Andhra Pradesh"
    
    def _clean_document_title(self, source_document: str) -> str:
        """Clean and format document title"""
        # Remove file extensions
        title = re.sub(r'\.(pdf|doc|docx|xlsx?|txt)$', '', source_document, flags=re.IGNORECASE)
        
        # Replace underscores and hyphens with spaces
        title = re.sub(r'[_-]', ' ', title)
        
        # Capitalize words
        title = ' '.join(word.capitalize() for word in title.split())
        
        return title
    
    # Citation formatters for different styles
    
    def _format_apa(self, metadata: SourceMetadata, page_ref: Optional[str] = None) -> Tuple[str, str]:
        """Format citation in APA style"""
        
        # Authors
        if metadata.authors:
            if len(metadata.authors) == 1:
                author_part = metadata.authors[0]
            elif len(metadata.authors) == 2:
                author_part = f"{metadata.authors[0]} & {metadata.authors[1]}"
            else:
                author_part = f"{metadata.authors[0]} et al."
        else:
            author_part = metadata.organization
        
        # Year
        year_part = f"({metadata.publication_date.year})" if metadata.publication_date else "(n.d.)"
        
        # Title
        title_part = f"{metadata.title}"
        
        # Organization (if not used as author)
        org_part = ""
        if metadata.authors and metadata.organization:
            org_part = f" {metadata.organization}."
        
        # URL and access date
        url_part = ""
        if metadata.url:
            url_part = f" Retrieved {metadata.access_date.strftime('%B %d, %Y')}, from {metadata.url}"
        
        # Full citation
        full_citation = f"{author_part} {year_part}. {title_part}.{org_part}{url_part}"
        
        # In-text citation
        year_str = str(metadata.publication_date.year) if metadata.publication_date else "n.d."
        in_text = f"({author_part.split()[0] if metadata.authors else metadata.organization}, {year_str}"
        if page_ref:
            in_text += f", p. {page_ref}"
        in_text += ")"
        
        return full_citation, in_text
    
    def _format_mla(self, metadata: SourceMetadata, page_ref: Optional[str] = None) -> Tuple[str, str]:
        """Format citation in MLA style"""
        
        # Authors
        if metadata.authors:
            author_part = metadata.authors[0] if len(metadata.authors) == 1 else f"{metadata.authors[0]} et al."
        else:
            author_part = metadata.organization
        
        # Title in quotes
        title_part = f'"{metadata.title}"'
        
        # Organization
        org_part = metadata.organization if metadata.authors else ""
        
        # Date
        if metadata.publication_date:
            date_part = metadata.publication_date.strftime('%d %b %Y')
        else:
            date_part = "n.d."
        
        # Web access
        web_part = ""
        if metadata.url:
            web_part = f" Web. {metadata.access_date.strftime('%d %b %Y')}"
        
        # Full citation
        parts = [author_part, title_part]
        if org_part and org_part != author_part:
            parts.append(org_part)
        parts.append(f"{date_part}.{web_part}")
        
        full_citation = ". ".join(parts)
        
        # In-text citation
        in_text = f"({author_part.split()[0]}"
        if page_ref:
            in_text += f" {page_ref}"
        in_text += ")"
        
        return full_citation, in_text
    
    def _format_chicago(self, metadata: SourceMetadata, page_ref: Optional[str] = None) -> Tuple[str, str]:
        """Format citation in Chicago style"""
        
        # Authors
        if metadata.authors:
            author_part = metadata.authors[0] if len(metadata.authors) == 1 else f"{metadata.authors[0]} et al."
        else:
            author_part = metadata.organization
        
        # Title in quotes
        title_part = f'"{metadata.title}"'
        
        # Publication info
        pub_info = metadata.organization
        if metadata.publication_date:
            pub_info += f", {metadata.publication_date.year}"
        
        # URL and access date
        url_part = ""
        if metadata.url:
            url_part = f" Accessed {metadata.access_date.strftime('%B %d, %Y')}. {metadata.url}."
        
        # Full citation (footnote style)
        full_citation = f"{author_part}, {title_part} ({pub_info}){url_part}"
        
        # In-text citation (author-date style)
        year_str = str(metadata.publication_date.year) if metadata.publication_date else "n.d."
        in_text = f"({author_part.split()[0]} {year_str}"
        if page_ref:
            in_text += f", {page_ref}"
        in_text += ")"
        
        return full_citation, in_text
    
    def _format_ieee(self, metadata: SourceMetadata, page_ref: Optional[str] = None) -> Tuple[str, str]:
        """Format citation in IEEE style"""
        
        # Authors
        if metadata.authors:
            author_part = ", ".join(metadata.authors[:3])
            if len(metadata.authors) > 3:
                author_part += " et al."
        else:
            author_part = metadata.organization
        
        # Title in quotes
        title_part = f'"{metadata.title}"'
        
        # Publication info
        pub_info = metadata.organization
        if metadata.publication_date:
            pub_info += f", {metadata.publication_date.year}"
        
        # URL and access date
        url_part = ""
        if metadata.url:
            url_part = f" [Online]. Available: {metadata.url}. [Accessed: {metadata.access_date.strftime('%d-%b-%Y')}]"
        
        # Full citation
        full_citation = f"{author_part}, {title_part}, {pub_info}.{url_part}"
        
        # In-text citation (numbered)
        in_text = "[1]"  # Simplified - would need numbering system
        
        return full_citation, in_text
    
    def _format_vancouver(self, metadata: SourceMetadata, page_ref: Optional[str] = None) -> Tuple[str, str]:
        """Format citation in Vancouver style"""
        
        # Authors (last name, initials)
        if metadata.authors:
            author_part = metadata.authors[0]
            if len(metadata.authors) > 1:
                author_part += " et al"
        else:
            author_part = metadata.organization
        
        # Title
        title_part = metadata.title
        
        # Publication info
        pub_info = metadata.organization
        if metadata.publication_date:
            pub_info += f"; {metadata.publication_date.year}"
        
        # URL and access date
        url_part = ""
        if metadata.url:
            url_part = f" [cited {metadata.access_date.strftime('%Y %b %d')}]. Available from: {metadata.url}"
        
        # Full citation
        full_citation = f"{author_part}. {title_part}. {pub_info}.{url_part}"
        
        # In-text citation (numbered)
        in_text = "(1)"  # Simplified
        
        return full_citation, in_text
    
    def _format_harvard(self, metadata: SourceMetadata, page_ref: Optional[str] = None) -> Tuple[str, str]:
        """Format citation in Harvard style"""
        
        # Authors
        if metadata.authors:
            if len(metadata.authors) == 1:
                author_part = metadata.authors[0]
            else:
                author_part = f"{metadata.authors[0]} et al."
        else:
            author_part = metadata.organization
        
        # Year
        year_part = str(metadata.publication_date.year) if metadata.publication_date else "n.d."
        
        # Title in italics (represented with underscores)
        title_part = f"_{metadata.title}_"
        
        # Publication info
        pub_info = metadata.organization
        
        # URL and access date
        url_part = ""
        if metadata.url:
            url_part = f" Available at: {metadata.url} (Accessed: {metadata.access_date.strftime('%d %B %Y')})"
        
        # Full citation
        full_citation = f"{author_part}, {year_part}, {title_part}, {pub_info}.{url_part}"
        
        # In-text citation
        in_text = f"({author_part.split()[0]}, {year_part}"
        if page_ref:
            in_text += f", p.{page_ref}"
        in_text += ")"
        
        return full_citation, in_text
    
    def get_bibliography(
        self,
        citation_ids: List[str],
        style: CitationStyle = CitationStyle.APA,
        sort_alphabetically: bool = True
    ) -> List[str]:
        """Generate bibliography from citation IDs"""
        try:
            citations = []
            
            for citation_id in citation_ids:
                if citation_id in self.citations_cache:
                    citation = self.citations_cache[citation_id]
                    # Re-format if different style requested
                    if citation.style != style:
                        formatter = self.style_formatters.get(style, self._format_apa)
                        formatted_citation, _ = formatter(citation.source_metadata, citation.page_reference)
                        citations.append(formatted_citation)
                    else:
                        citations.append(citation.formatted_citation)
            
            if sort_alphabetically:
                citations.sort()
            
            return citations
            
        except Exception as e:
            logger.error(f"Failed to generate bibliography: {e}")
            return []
    
    def export_citations(
        self,
        format_type: str = "json",
        file_path: Optional[str] = None
    ) -> Optional[str]:
        """Export citations to file"""
        try:
            if format_type.lower() == "json":
                data = {
                    'sources': {k: asdict(v) for k, v in self.sources_cache.items()},
                    'citations': {k: asdict(v) for k, v in self.citations_cache.items()}
                }
                
                # Convert datetime objects to strings
                def convert_datetime(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, date):
                        return obj.isoformat()
                    return obj
                
                # Process the data to convert datetime objects
                def process_dict(d):
                    if isinstance(d, dict):
                        return {k: process_dict(v) for k, v in d.items()}
                    elif isinstance(d, list):
                        return [process_dict(item) for item in d]
                    else:
                        return convert_datetime(d)
                
                processed_data = process_dict(data)
                
                if file_path:
                    with open(file_path, 'w') as f:
                        json.dump(processed_data, f, indent=2, default=str)
                    return file_path
                else:
                    return json.dumps(processed_data, indent=2, default=str)
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            logger.error(f"Failed to export citations: {e}")
            return None
    
    def _load_citations(self):
        """Load citations from database file"""
        try:
            if Path(self.citation_database_path).exists():
                with open(self.citation_database_path, 'r') as f:
                    data = json.load(f)
                
                # Load sources
                sources_data = data.get('sources', {})
                for source_id, source_dict in sources_data.items():
                    # Convert date strings back to date objects
                    if source_dict.get('publication_date'):
                        source_dict['publication_date'] = datetime.fromisoformat(source_dict['publication_date']).date()
                    if source_dict.get('access_date'):
                        source_dict['access_date'] = datetime.fromisoformat(source_dict['access_date']).date()
                    
                    self.sources_cache[source_id] = SourceMetadata(**source_dict)
                
                # Load citations
                citations_data = data.get('citations', {})
                for citation_id, citation_dict in citations_data.items():
                    # Reconstruct citation object
                    source_metadata = self.sources_cache[citation_dict['source_metadata']['source_id']]
                    
                    citation = Citation(
                        citation_id=citation_dict['citation_id'],
                        source_metadata=source_metadata,
                        style=CitationStyle(citation_dict['style']),
                        formatted_citation=citation_dict['formatted_citation'],
                        in_text_citation=citation_dict['in_text_citation'],
                        created_at=datetime.fromisoformat(citation_dict['created_at']),
                        page_reference=citation_dict.get('page_reference'),
                        quote_text=citation_dict.get('quote_text'),
                        context=citation_dict.get('context')
                    )
                    
                    self.citations_cache[citation_id] = citation
                
                logger.info(f"Loaded {len(self.sources_cache)} sources and {len(self.citations_cache)} citations")
                
        except Exception as e:
            logger.warning(f"Could not load citations database: {e}")
            # Initialize empty caches
            self.sources_cache = {}
            self.citations_cache = {}
    
    def _save_citations(self):
        """Save citations to database file"""
        try:
            # Ensure directory exists
            Path(self.citation_database_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export to file
            self.export_citations(format_type="json", file_path=self.citation_database_path)
            
        except Exception as e:
            logger.error(f"Failed to save citations database: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get citation service statistics"""
        return {
            'total_sources': len(self.sources_cache),
            'total_citations': len(self.citations_cache),
            'source_types': {
                source_type.value: sum(1 for s in self.sources_cache.values() if s.source_type == source_type)
                for source_type in SourceType
            },
            'citation_styles': {
                style.value: sum(1 for c in self.citations_cache.values() if c.style == style)
                for style in CitationStyle
            },
            'database_path': self.citation_database_path
        }