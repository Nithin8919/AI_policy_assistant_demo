#!/usr/bin/env python3
"""
Enhanced Legal Document Processor
Adds GO supersession tracking, judgment citation parsing, and legal hierarchy extraction
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class LegalReference:
    """Represents a legal reference or citation"""
    ref_type: str  # 'act', 'go', 'judgment', 'rule', 'section'
    ref_id: str    # GO number, case citation, act section, etc.
    title: str     # Full title or description
    year: Optional[int] = None
    source_text: str = ""
    confidence: float = 0.0

@dataclass
class LegalHierarchy:
    """Represents hierarchical structure in legal documents"""
    level: int           # 0=document, 1=part, 2=chapter, 3=section, 4=subsection
    identifier: str      # Section number, article number, etc.
    title: str          # Section title
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    content: str = ""

@dataclass
class GOSupersession:
    """Tracks Government Order supersession chains"""
    superseding_go: str      # New GO that supersedes
    superseded_go: str       # Old GO being superseded
    supersession_date: Optional[datetime] = None
    supersession_text: str = ""
    supersession_type: str = "full"  # 'full', 'partial', 'amendment'

class EnhancedLegalProcessor:
    """Enhanced processor for complex legal document structures"""
    
    def __init__(self):
        # Enhanced legal patterns
        self.legal_patterns = {
            # Government Orders
            'go_reference': re.compile(
                r'G\.?O\.?\s*(?:No\.?)?\s*(\d+)(?:/(\d{4}))?|'
                r'Government\s+Order\s+(?:No\.?)?\s*(\d+)(?:/(\d{4}))?',
                re.IGNORECASE
            ),
            'go_supersession': re.compile(
                r'(?:hereby\s+supersedes?|is\s+superseded\s+by|replaces?|'
                r'cancels?\s+and\s+supersedes?)\s+.*?G\.?O\.?\s*(?:No\.?)?\s*(\d+)(?:/(\d{4}))?',
                re.IGNORECASE | re.DOTALL
            ),
            
            # Acts and Rules
            'act_reference': re.compile(
                r'(?:The\s+)?([A-Z][A-Za-z\s]+(?:Act|Rule))\s*,?\s*(\d{4})',
                re.IGNORECASE
            ),
            'section_reference': re.compile(
                r'Section\s+(\d+(?:\([a-z]\))?(?:\([i-v]+\))?)',
                re.IGNORECASE
            ),
            'rule_reference': re.compile(
                r'Rule\s+(\d+(?:\([a-z]\))?(?:\([i-v]+\))?)',
                re.IGNORECASE
            ),
            
            # Judgments and Citations
            'case_citation': re.compile(
                r'(?:(\d{4})\s+)?(\w+)\s+(\d+)\s+(SC|HC|SCC|AIR|WLR)',
                re.IGNORECASE
            ),
            'court_reference': re.compile(
                r'(?:Supreme\s+Court|High\s+Court|District\s+Court|'
                r'Appellate\s+Tribunal|CESTAT|CAT)',
                re.IGNORECASE
            ),
            
            # Legal Hierarchy
            'chapter_header': re.compile(
                r'^CHAPTER\s+([IVX]+|\d+)[\s\-:]*(.+)$',
                re.MULTILINE | re.IGNORECASE
            ),
            'part_header': re.compile(
                r'^PART\s+([IVX]+|\d+)[\s\-:]*(.+)$',
                re.MULTILINE | re.IGNORECASE
            ),
            'section_header': re.compile(
                r'^(\d+)\.?\s+(.+)$',
                re.MULTILINE
            ),
            'subsection_header': re.compile(
                r'^\((\d+|[a-z])\)\s+(.+)$',
                re.MULTILINE
            ),
            
            # Amendment patterns
            'amendment': re.compile(
                r'(?:as\s+amended\s+by|amended\s+vide|substituted\s+by|'
                r'inserted\s+by|omitted\s+by)\s+.*?(\d{4})',
                re.IGNORECASE | re.DOTALL
            ),
            
            # Definition patterns
            'definition_clause': re.compile(
                r'"([^"]+)"\s+means?\s+(.+?)(?=\n\n|\n[A-Z]|$)',
                re.IGNORECASE | re.DOTALL
            ),
            
            # Procedure patterns
            'procedure_step': re.compile(
                r'(?:Step\s+(\d+)|Procedure\s+(\d+)|Process\s+(\d+)):\s*(.+)',
                re.IGNORECASE
            )
        }
        
        # Legal entity patterns
        self.legal_entities = {
            'GOVERNMENT_DEPARTMENTS': [
                'Department of School Education', 'CSE', 'SCERT', 'SIEMAT',
                'DIET', 'BRC', 'CRC', 'RMSA', 'SSA', 'MDM'
            ],
            'EDUCATION_ACTS': [
                'Right to Education Act', 'RTE Act', 'Education Act',
                'Children\'s Act', 'POCSO Act', 'Juvenile Justice Act'
            ],
            'EDUCATION_POLICIES': [
                'National Education Policy', 'NEP', 'State Education Policy',
                'Mid Day Meal Scheme', 'Sarva Shiksha Abhiyan'
            ],
            'COURTS': [
                'Supreme Court', 'High Court', 'District Court',
                'Sessions Court', 'Magistrate Court'
            ]
        }
        
    def extract_legal_references(self, text: str) -> List[LegalReference]:
        """Extract all legal references from text"""
        references = []
        
        # Extract GO references
        go_refs = self._extract_go_references(text)
        references.extend(go_refs)
        
        # Extract Act references
        act_refs = self._extract_act_references(text)
        references.extend(act_refs)
        
        # Extract judgment citations
        judgment_refs = self._extract_judgment_references(text)
        references.extend(judgment_refs)
        
        # Extract section references
        section_refs = self._extract_section_references(text)
        references.extend(section_refs)
        
        return references
    
    def _extract_go_references(self, text: str) -> List[LegalReference]:
        """Extract Government Order references"""
        references = []
        
        for match in self.legal_patterns['go_reference'].finditer(text):
            go_number = match.group(1) or match.group(3)
            year = match.group(2) or match.group(4)
            
            ref_id = f"GO {go_number}"
            if year:
                ref_id += f"/{year}"
                
            references.append(LegalReference(
                ref_type='go',
                ref_id=ref_id,
                title=f"Government Order {ref_id}",
                year=int(year) if year else None,
                source_text=match.group(0),
                confidence=0.9
            ))
        
        return references
    
    def _extract_act_references(self, text: str) -> List[LegalReference]:
        """Extract Act and Rule references"""
        references = []
        
        for match in self.legal_patterns['act_reference'].finditer(text):
            act_name = match.group(1).strip()
            year = match.group(2)
            
            ref_id = f"{act_name}, {year}"
            
            references.append(LegalReference(
                ref_type='act',
                ref_id=ref_id,
                title=act_name,
                year=int(year),
                source_text=match.group(0),
                confidence=0.85
            ))
        
        return references
    
    def _extract_judgment_references(self, text: str) -> List[LegalReference]:
        """Extract court judgment citations"""
        references = []
        
        for match in self.legal_patterns['case_citation'].finditer(text):
            year = match.group(1)
            volume = match.group(2)
            page = match.group(3)
            reporter = match.group(4)
            
            ref_id = f"{year} {volume} {page} {reporter}" if year else f"{volume} {page} {reporter}"
            
            references.append(LegalReference(
                ref_type='judgment',
                ref_id=ref_id,
                title=f"Case Citation: {ref_id}",
                year=int(year) if year else None,
                source_text=match.group(0),
                confidence=0.8
            ))
        
        return references
    
    def _extract_section_references(self, text: str) -> List[LegalReference]:
        """Extract section and rule references"""
        references = []
        
        # Section references
        for match in self.legal_patterns['section_reference'].finditer(text):
            section_num = match.group(1)
            
            references.append(LegalReference(
                ref_type='section',
                ref_id=f"Section {section_num}",
                title=f"Section {section_num}",
                source_text=match.group(0),
                confidence=0.7
            ))
        
        # Rule references
        for match in self.legal_patterns['rule_reference'].finditer(text):
            rule_num = match.group(1)
            
            references.append(LegalReference(
                ref_type='rule',
                ref_id=f"Rule {rule_num}",
                title=f"Rule {rule_num}",
                source_text=match.group(0),
                confidence=0.7
            ))
        
        return references
    
    def extract_go_supersessions(self, text: str) -> List[GOSupersession]:
        """Extract GO supersession information"""
        supersessions = []
        
        for match in self.legal_patterns['go_supersession'].finditer(text):
            superseded_go_num = match.group(1)
            superseded_year = match.group(2)
            
            superseded_go = f"GO {superseded_go_num}"
            if superseded_year:
                superseded_go += f"/{superseded_year}"
            
            # Try to find the superseding GO in the same text
            superseding_go = self._find_current_go(text)
            
            if superseding_go:
                supersessions.append(GOSupersession(
                    superseding_go=superseding_go,
                    superseded_go=superseded_go,
                    supersession_text=match.group(0),
                    supersession_type=self._classify_supersession_type(match.group(0))
                ))
        
        return supersessions
    
    def _find_current_go(self, text: str) -> Optional[str]:
        """Find the current GO number in the document"""
        # Look for GO in document header/title
        lines = text.split('\n')[:10]  # Check first 10 lines
        
        for line in lines:
            match = self.legal_patterns['go_reference'].search(line)
            if match:
                go_number = match.group(1) or match.group(3)
                year = match.group(2) or match.group(4)
                
                ref_id = f"GO {go_number}"
                if year:
                    ref_id += f"/{year}"
                return ref_id
        
        return None
    
    def _classify_supersession_type(self, supersession_text: str) -> str:
        """Classify the type of supersession"""
        text_lower = supersession_text.lower()
        
        if any(word in text_lower for word in ['partially', 'part', 'certain sections']):
            return 'partial'
        elif any(word in text_lower for word in ['amend', 'modify', 'substitute']):
            return 'amendment'
        else:
            return 'full'
    
    def extract_legal_hierarchy(self, text: str) -> List[LegalHierarchy]:
        """Extract hierarchical structure from legal documents"""
        hierarchy = []
        current_parents = {}  # Track current parent at each level
        
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for different hierarchy levels
            hierarchy_item = None
            
            # Part headers (Level 1)
            part_match = self.legal_patterns['part_header'].match(line)
            if part_match:
                part_num, part_title = part_match.groups()
                hierarchy_item = LegalHierarchy(
                    level=1,
                    identifier=f"Part {part_num}",
                    title=part_title.strip(),
                    content=line
                )
                current_parents[1] = hierarchy_item.identifier
                current_parents = {k: v for k, v in current_parents.items() if k <= 1}
            
            # Chapter headers (Level 2)
            elif self.legal_patterns['chapter_header'].match(line):
                chapter_match = self.legal_patterns['chapter_header'].match(line)
                chapter_num, chapter_title = chapter_match.groups()
                hierarchy_item = LegalHierarchy(
                    level=2,
                    identifier=f"Chapter {chapter_num}",
                    title=chapter_title.strip(),
                    parent_id=current_parents.get(1),
                    content=line
                )
                current_parents[2] = hierarchy_item.identifier
                current_parents = {k: v for k, v in current_parents.items() if k <= 2}
            
            # Section headers (Level 3)
            elif self.legal_patterns['section_header'].match(line):
                section_match = self.legal_patterns['section_header'].match(line)
                section_num, section_title = section_match.groups()
                hierarchy_item = LegalHierarchy(
                    level=3,
                    identifier=f"Section {section_num}",
                    title=section_title.strip(),
                    parent_id=current_parents.get(2),
                    content=line
                )
                current_parents[3] = hierarchy_item.identifier
                current_parents = {k: v for k, v in current_parents.items() if k <= 3}
            
            # Subsection headers (Level 4)
            elif self.legal_patterns['subsection_header'].match(line):
                subsection_match = self.legal_patterns['subsection_header'].match(line)
                subsection_num, subsection_title = subsection_match.groups()
                hierarchy_item = LegalHierarchy(
                    level=4,
                    identifier=f"Subsection ({subsection_num})",
                    title=subsection_title.strip(),
                    parent_id=current_parents.get(3),
                    content=line
                )
            
            if hierarchy_item:
                hierarchy.append(hierarchy_item)
                
                # Update parent-child relationships
                if hierarchy_item.parent_id:
                    for parent in hierarchy:
                        if parent.identifier == hierarchy_item.parent_id:
                            parent.children.append(hierarchy_item.identifier)
                            break
        
        return hierarchy
    
    def extract_definitions(self, text: str) -> Dict[str, str]:
        """Extract legal definitions"""
        definitions = {}
        
        for match in self.legal_patterns['definition_clause'].finditer(text):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            
            # Clean up definition text
            definition = re.sub(r'\s+', ' ', definition)
            definitions[term] = definition
        
        return definitions
    
    def extract_procedures(self, text: str) -> List[Dict[str, Any]]:
        """Extract procedural steps"""
        procedures = []
        
        for match in self.legal_patterns['procedure_step'].finditer(text):
            step_num = match.group(1) or match.group(2) or match.group(3)
            step_desc = match.group(4).strip()
            
            procedures.append({
                'step_number': int(step_num) if step_num.isdigit() else step_num,
                'description': step_desc,
                'source_text': match.group(0)
            })
        
        return procedures
    
    def analyze_legal_document(self, text: str, doc_id: str) -> Dict[str, Any]:
        """Comprehensive analysis of a legal document"""
        
        analysis = {
            'document_id': doc_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'document_type': self._classify_document_type(text),
            'legal_references': self.extract_legal_references(text),
            'go_supersessions': self.extract_go_supersessions(text),
            'legal_hierarchy': self.extract_legal_hierarchy(text),
            'definitions': self.extract_definitions(text),
            'procedures': self.extract_procedures(text),
            'amendments': self._extract_amendments(text),
            'effective_dates': self._extract_effective_dates(text),
            'jurisdiction': self._extract_jurisdiction(text)
        }
        
        # Calculate document complexity score
        analysis['complexity_score'] = self._calculate_complexity_score(analysis)
        
        return analysis
    
    def _classify_document_type(self, text: str) -> str:
        """Classify the type of legal document"""
        text_lower = text.lower()
        
        if 'government order' in text_lower or 'g.o.' in text_lower:
            return 'government_order'
        elif any(word in text_lower for word in ['act', 'statute']):
            return 'act'
        elif 'rule' in text_lower and 'act' not in text_lower:
            return 'rule'
        elif any(word in text_lower for word in ['judgment', 'order', 'writ']):
            return 'judgment'
        elif 'policy' in text_lower:
            return 'policy'
        elif 'circular' in text_lower:
            return 'circular'
        elif 'notification' in text_lower:
            return 'notification'
        else:
            return 'unknown'
    
    def _extract_amendments(self, text: str) -> List[Dict[str, Any]]:
        """Extract amendment information"""
        amendments = []
        
        for match in self.legal_patterns['amendment'].finditer(text):
            year = match.group(1)
            
            amendments.append({
                'type': 'amendment',
                'year': int(year),
                'text': match.group(0),
                'amendment_type': self._classify_amendment_type(match.group(0))
            })
        
        return amendments
    
    def _classify_amendment_type(self, amendment_text: str) -> str:
        """Classify type of amendment"""
        text_lower = amendment_text.lower()
        
        if 'substituted' in text_lower:
            return 'substitution'
        elif 'inserted' in text_lower:
            return 'insertion'
        elif 'omitted' in text_lower:
            return 'omission'
        elif 'amended' in text_lower:
            return 'amendment'
        else:
            return 'modification'
    
    def _extract_effective_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract effective dates"""
        date_patterns = [
            r'effective\s+from\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'comes?\s+into\s+force\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
            r'w\.?e\.?f\.?\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})'
        ]
        
        dates = []
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                dates.append({
                    'date': match.group(1),
                    'context': match.group(0),
                    'type': 'effective_date'
                })
        
        return dates
    
    def _extract_jurisdiction(self, text: str) -> Dict[str, Any]:
        """Extract jurisdiction information"""
        jurisdiction = {
            'state': None,
            'district': None,
            'department': None,
            'scope': 'unknown'
        }
        
        # Check for Andhra Pradesh
        if re.search(r'andhra\s+pradesh|a\.?p\.?(?:\s+government)?', text, re.IGNORECASE):
            jurisdiction['state'] = 'Andhra Pradesh'
        
        # Check for specific districts
        ap_districts = [
            'anantapur', 'chittoor', 'east godavari', 'guntur', 'kadapa',
            'krishna', 'kurnool', 'nellore', 'prakasam', 'srikakulam',
            'visakhapatnam', 'vizianagaram', 'west godavari'
        ]
        
        for district in ap_districts:
            if district in text.lower():
                jurisdiction['district'] = district.title()
                break
        
        # Check for education department
        if any(term in text.lower() for term in ['school education', 'education department', 'cse']):
            jurisdiction['department'] = 'School Education'
        
        # Determine scope
        if jurisdiction['district']:
            jurisdiction['scope'] = 'district'
        elif jurisdiction['state']:
            jurisdiction['scope'] = 'state'
        else:
            jurisdiction['scope'] = 'unknown'
        
        return jurisdiction
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate document complexity score (0-1)"""
        score = 0.0
        
        # References complexity
        score += min(len(analysis['legal_references']) * 0.1, 0.3)
        
        # Hierarchy complexity
        score += min(len(analysis['legal_hierarchy']) * 0.05, 0.2)
        
        # Definitions complexity
        score += min(len(analysis['definitions']) * 0.05, 0.1)
        
        # Procedures complexity
        score += min(len(analysis['procedures']) * 0.03, 0.1)
        
        # Supersessions complexity
        score += min(len(analysis['go_supersessions']) * 0.1, 0.2)
        
        # Base complexity for document type
        doc_type = analysis['document_type']
        if doc_type == 'act':
            score += 0.3
        elif doc_type == 'government_order':
            score += 0.2
        elif doc_type == 'judgment':
            score += 0.25
        else:
            score += 0.1
        
        return min(score, 1.0)


def main():
    """Test the enhanced legal processor"""
    
    # Sample legal text
    sample_text = """
    Government Order No. 45/2023
    
    This order hereby supersedes G.O. No. 12/2019 dated 15th March 2019.
    
    Section 12. School Management Committees
    (a) Every school shall constitute a School Management Committee
    (b) The SMC shall consist of elected representatives
    
    "School Management Committee" means the committee constituted under Section 21 of the RTE Act, 2009.
    
    This order comes into force w.e.f. 1st April 2023.
    """
    
    processor = EnhancedLegalProcessor()
    analysis = processor.analyze_legal_document(sample_text, "GO_45_2023")
    
    print("Enhanced Legal Analysis Results:")
    print(f"Document Type: {analysis['document_type']}")
    print(f"Legal References: {len(analysis['legal_references'])}")
    print(f"GO Supersessions: {len(analysis['go_supersessions'])}")
    print(f"Hierarchy Items: {len(analysis['legal_hierarchy'])}")
    print(f"Definitions: {len(analysis['definitions'])}")
    print(f"Complexity Score: {analysis['complexity_score']:.3f}")
    
    # Print supersessions
    for supersession in analysis['go_supersessions']:
        print(f"Supersession: {supersession.superseding_go} supersedes {supersession.superseded_go}")


if __name__ == "__main__":
    main()