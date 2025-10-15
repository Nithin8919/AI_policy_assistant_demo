#!/usr/bin/env python3
"""
Legal Document Processor for AP Policy Co-Pilot
Handles Acts, Rules, GOs with clause extraction, supersession tracking, and hierarchy building
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from enum import Enum

logger = logging.getLogger(__name__)

class LegalDocType(Enum):
    ACT = "act"
    RULE = "rule"
    GO = "go"
    CIRCULAR = "circular"
    NOTIFICATION = "notification"
    JUDGMENT = "judgment"

@dataclass
class LegalClause:
    """Structured legal clause"""
    clause_id: str
    clause_number: str
    clause_text: str
    parent_section: Optional[str] = None
    sub_clauses: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
@dataclass
class LegalDocument:
    """Structured legal document"""
    doc_id: str
    doc_type: LegalDocType
    title: str
    full_text: str
    sections: Dict[str, str] = field(default_factory=dict)
    clauses: List[LegalClause] = field(default_factory=list)
    
    # Metadata
    issued_by: Optional[str] = None
    issued_date: Optional[str] = None
    effective_date: Optional[str] = None
    go_number: Optional[str] = None
    department: Optional[str] = None
    
    # Supersession tracking
    supersedes: List[str] = field(default_factory=list)  # List of doc_ids
    superseded_by: Optional[str] = None
    amendments: List[str] = field(default_factory=list)
    
    # Hierarchy
    parent_act: Optional[str] = None
    parent_rule: Optional[str] = None
    
    # Source
    source_file: Optional[str] = None
    source_url: Optional[str] = None
    
    # Processing metadata
    extraction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0

class LegalDocumentProcessor:
    """
    Processes legal documents with:
    1. Clause-level extraction
    2. Supersession tracking
    3. Legal hierarchy building
    4. Citation extraction
    """
    
    def __init__(self, corpus_index_path: str):
        self.corpus_index_path = Path(corpus_index_path)
        self.corpus_index = None
        self.documents = {}
        self.hierarchy = {}
        
        self._load_corpus_index()
    
    def _load_corpus_index(self):
        """Load corpus index CSV"""
        if not self.corpus_index_path.exists():
            logger.warning(f"Corpus index not found: {self.corpus_index_path}")
            self.corpus_index = pd.DataFrame(columns=[
                'doc_id', 'title', 'type', 'issuer', 'jurisdiction', 
                'effective_date', 'pub_date', 'url', 'file_path', 'version', 'notes'
            ])
        else:
            self.corpus_index = pd.read_csv(self.corpus_index_path)
            logger.info(f"Loaded {len(self.corpus_index)} documents from corpus index")
    
    def process_document(self, file_path: str, doc_type: LegalDocType) -> LegalDocument:
        """Main processing pipeline for a legal document"""
        logger.info(f"Processing {doc_type.value}: {file_path}")
        
        # Extract text
        text = self._extract_text(file_path)
        
        # Parse based on document type
        if doc_type == LegalDocType.GO:
            return self._process_go(text, file_path)
        elif doc_type == LegalDocType.ACT:
            return self._process_act(text, file_path)
        elif doc_type == LegalDocType.RULE:
            return self._process_rule(text, file_path)
        else:
            return self._process_generic(text, file_path, doc_type)
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from PDF/TXT"""
        import PyPDF2
        
        path = Path(file_path)
        if path.suffix == '.pdf':
            with open(path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif path.suffix == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        return text
    
    def _process_go(self, text: str, file_path: str) -> LegalDocument:
        """Process Government Order"""
        doc = LegalDocument(
            doc_id=self._generate_doc_id(),
            doc_type=LegalDocType.GO,
            title=self._extract_title(text),
            full_text=text,
            source_file=file_path
        )
        
        # Extract GO number
        go_number_match = re.search(r'G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)', text, re.IGNORECASE)
        if go_number_match:
            doc.go_number = go_number_match.group(0)
        
        # Extract department
        dept_patterns = [
            r'(?:From|Department)[\s:]+([^\n]+(?:Department|Directorate|Ministry)[^\n]*)',
            r'([A-Z][A-Za-z\s&]+(?:Department|Directorate))'
        ]
        for pattern in dept_patterns:
            dept_match = re.search(pattern, text)
            if dept_match:
                doc.department = dept_match.group(1).strip()
                break
        
        # Extract dates
        date_patterns = [
            r'Dated[\s:]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'Date[\s:]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, text)
            if date_match:
                doc.issued_date = self._normalize_date(date_match.group(1))
                break
        
        # Extract supersession information
        supersession_patterns = [
            r'(?:hereby )?(?:supersede|repeal|cancel|rescind)[s]?\s+G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)',
            r'in (?:supersession|place) of G\.?O\.?\s*(?:Ms\.?|Rt\.?)?\s*No\.?\s*(\d+)'
        ]
        for pattern in supersession_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                doc.supersedes.append(f"GO_{match.group(1)}")
        
        # Extract sections/clauses
        doc.sections = self._extract_sections(text)
        doc.clauses = self._extract_clauses_from_go(text)
        
        # Extract references to Acts/Rules
        doc.parent_act = self._extract_parent_act(text)
        doc.parent_rule = self._extract_parent_rule(text)
        
        return doc
    
    def _process_act(self, text: str, file_path: str) -> LegalDocument:
        """Process Act"""
        doc = LegalDocument(
            doc_id=self._generate_doc_id(),
            doc_type=LegalDocType.ACT,
            title=self._extract_title(text),
            full_text=text,
            source_file=file_path
        )
        
        # Extract sections
        doc.sections = self._extract_sections_from_act(text)
        
        # Extract clauses
        doc.clauses = self._extract_clauses_from_act(text)
        
        # Extract enactment date
        enactment_match = re.search(r'(?:enacted|passed).*?(\d{4})', text, re.IGNORECASE)
        if enactment_match:
            doc.issued_date = enactment_match.group(1)
        
        return doc
    
    def _process_rule(self, text: str, file_path: str) -> LegalDocument:
        """Process Rule"""
        doc = LegalDocument(
            doc_id=self._generate_doc_id(),
            doc_type=LegalDocType.RULE,
            title=self._extract_title(text),
            full_text=text,
            source_file=file_path
        )
        
        # Extract parent Act
        doc.parent_act = self._extract_parent_act(text)
        
        # Extract sections
        doc.sections = self._extract_sections(text)
        doc.clauses = self._extract_clauses_from_rule(text)
        
        return doc
    
    def _process_generic(self, text: str, file_path: str, doc_type: LegalDocType) -> LegalDocument:
        """Process generic legal document"""
        return LegalDocument(
            doc_id=self._generate_doc_id(),
            doc_type=doc_type,
            title=self._extract_title(text),
            full_text=text,
            source_file=file_path,
            sections=self._extract_sections(text)
        )
    
    def _extract_title(self, text: str) -> str:
        """Extract document title"""
        lines = text.split('\n')
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if len(line) > 10 and line.isupper():
                return line
        return "Untitled Document"
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from document"""
        sections = {}
        pattern = r'(?:Section|Article|Chapter)\s+(\d+[A-Z]?)[\s:.-]+([^\n]+(?:\n(?!(?:Section|Article|Chapter)\s+\d)[^\n]+)*)'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            section_num = match.group(1)
            section_text = match.group(2).strip()
            sections[section_num] = section_text
        
        return sections
    
    def _extract_sections_from_act(self, text: str) -> Dict[str, str]:
        """Extract sections from Act (more structured)"""
        sections = {}
        # Acts typically have numbered sections
        pattern = r'(\d+)\.\s+([^\n]+(?:\n(?!\d+\.)[^\n]+)*)'
        
        for match in re.finditer(pattern, text):
            section_num = match.group(1)
            section_text = match.group(2).strip()
            sections[section_num] = section_text
        
        return sections
    
    def _extract_clauses_from_go(self, text: str) -> List[LegalClause]:
        """Extract clauses from GO"""
        clauses = []
        # GOs often have numbered paragraphs or clauses
        pattern = r'(?:^|\n)(\d+)\.\s+([^\n]+(?:\n(?!\d+\.)[^\n]+)*)'
        
        for match in re.finditer(pattern, text):
            clause = LegalClause(
                clause_id=f"CLAUSE_{match.group(1)}",
                clause_number=match.group(1),
                clause_text=match.group(2).strip()
            )
            clauses.append(clause)
        
        return clauses
    
    def _extract_clauses_from_act(self, text: str) -> List[LegalClause]:
        """Extract clauses from Act"""
        clauses = []
        # Acts have sections with sub-clauses (a), (b), (c)
        pattern = r'\(([a-z])\)\s+([^\n]+(?:\n(?!\([a-z]\))[^\n]+)*)'
        
        for match in re.finditer(pattern, text):
            clause = LegalClause(
                clause_id=f"CLAUSE_{match.group(1)}",
                clause_number=match.group(1),
                clause_text=match.group(2).strip()
            )
            clauses.append(clause)
        
        return clauses
    
    def _extract_clauses_from_rule(self, text: str) -> List[LegalClause]:
        """Extract clauses from Rule"""
        return self._extract_clauses_from_act(text)  # Similar structure
    
    def _extract_parent_act(self, text: str) -> Optional[str]:
        """Extract parent Act reference"""
        pattern = r'([A-Z][A-Za-z\s,]+(?:Act|LAW),?\s*\d{4})'
        match = re.search(pattern, text)
        return match.group(1) if match else None
    
    def _extract_parent_rule(self, text: str) -> Optional[str]:
        """Extract parent Rule reference"""
        pattern = r'([A-Z][A-Za-z\s]+Rules?,?\s*\d{4})'
        match = re.search(pattern, text)
        return match.group(1) if match else None
    
    def _generate_doc_id(self) -> str:
        """Generate unique document ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date to ISO format"""
        from dateutil import parser
        try:
            dt = parser.parse(date_str, fuzzy=True)
            return dt.strftime('%Y-%m-%d')
        except:
            return date_str
    
    def build_legal_hierarchy(self):
        """Build Act → Rule → GO hierarchy"""
        logger.info("Building legal hierarchy...")
        
        # Group by document type
        acts = {doc_id: doc for doc_id, doc in self.documents.items() if doc.doc_type == LegalDocType.ACT}
        rules = {doc_id: doc for doc_id, doc in self.documents.items() if doc.doc_type == LegalDocType.RULE}
        gos = {doc_id: doc for doc_id, doc in self.documents.items() if doc.doc_type == LegalDocType.GO}
        
        # Link Rules to Acts
        for rule_id, rule in rules.items():
            if rule.parent_act:
                for act_id, act in acts.items():
                    if rule.parent_act.lower() in act.title.lower():
                        rule.parent_act = act_id
                        break
        
        # Link GOs to Rules/Acts
        for go_id, go in gos.items():
            if go.parent_rule:
                for rule_id, rule in rules.items():
                    if go.parent_rule.lower() in rule.title.lower():
                        go.parent_rule = rule_id
                        break
            
            if go.parent_act:
                for act_id, act in acts.items():
                    if go.parent_act.lower() in act.title.lower():
                        go.parent_act = act_id
                        break
        
        logger.info(f"Built hierarchy: {len(acts)} Acts, {len(rules)} Rules, {len(gos)} GOs")
    
    def track_supersessions(self):
        """Track document supersessions"""
        logger.info("Tracking supersessions...")
        
        for doc_id, doc in self.documents.items():
            for superseded_ref in doc.supersedes:
                # Find the superseded document
                for other_id, other_doc in self.documents.items():
                    if superseded_ref in other_doc.go_number or superseded_ref in other_id:
                        other_doc.superseded_by = doc_id
                        logger.info(f"{doc.title} supersedes {other_doc.title}")
    
    def verify_corpus(self) -> bool:
        """Verify legal corpus is loaded"""
        return len(self.corpus_index) > 0
    
    def get_document(self, doc_id: str) -> Optional[LegalDocument]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def get_active_documents(self) -> List[LegalDocument]:
        """Get all documents that are not superseded"""
        return [doc for doc in self.documents.values() if not doc.superseded_by]
    
    def export_to_weaviate_format(self, doc: LegalDocument) -> Dict[str, Any]:
        """Export document to Weaviate format"""
        return {
            "doc_id": doc.doc_id,
            "doc_type": doc.doc_type.value,
            "title": doc.title,
            "full_text": doc.full_text,
            "go_number": doc.go_number,
            "department": doc.department,
            "issued_date": doc.issued_date,
            "effective_date": doc.effective_date,
            "supersedes": doc.supersedes,
            "superseded_by": doc.superseded_by,
            "parent_act": doc.parent_act,
            "parent_rule": doc.parent_rule,
            "source_file": doc.source_file,
            "source_url": doc.source_url,
            "confidence": doc.confidence
        }

if __name__ == "__main__":
    # Test processing
    processor = LegalDocumentProcessor("data/corpus_index.csv")
    
    # Example: Process a GO
    doc = processor.process_document("data/legal_frameworks/nadu_nedu_go.pdf", LegalDocType.GO)
    print(f"Processed: {doc.title}")
    print(f"GO Number: {doc.go_number}")
    print(f"Department: {doc.department}")
    print(f"Sections: {len(doc.sections)}")
    print(f"Clauses: {len(doc.clauses)}")