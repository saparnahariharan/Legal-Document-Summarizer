"""
Legal Information Extractor Module
Extracts vital legal provisions, entities, and linguistic constructs from legal documents:
1. Standard Clause Categorization (Termination, Confidentiality, Governing Law, Liability, etc.)
2. Dates and Timeframes (Execution dates, effective dates, notice periods)
3. Monetary Amounts and Currencies ($, €, £, ₹, USD, written amounts)
4. Legal Obligations ("shall", "must", "is required to")
5. Legal Conditions ("if", "provided that", "in the event of")
6. Legal Exceptions and Carve-Outs ("except", "unless", "notwithstanding")
"""

import re
from typing import List, Dict, Any


class LegalInformationExtractor:
    """Extracts structured legal clauses, entities, and syntactic markers."""

    # Pre-defined legal clause categories and their keyword patterns
    CLAUSE_DEFINITIONS = {
        'Termination': {
            'title': 'Term & Termination',
            'badge_color': 'danger',
            'patterns': [
                r'\bterminat(e|ion|ed|ing)\b', r'\bterm of (this|the) agreement\b',
                r'\bright to cancel\b', r'\bexpiration\b', r'\bnotice of termination\b',
                r'\bsurvive[s]? termination\b'
            ]
        },
        'Confidentiality': {
            'title': 'Confidentiality & Non-Disclosure',
            'badge_color': 'info',
            'patterns': [
                r'\bconfidential(ity)?\b', r'\bnon-disclosure\b', r'\bproprietary information\b',
                r'\btrade secret[s]?\b', r'\bkeep confidential\b', r'\bunauthorized disclosure\b'
            ]
        },
        'Governing Law': {
            'title': 'Governing Law & Jurisdiction',
            'badge_color': 'primary',
            'patterns': [
                r'\bgovern(ed)? by the laws?\b', r'\bjurisdiction\b', r'\bvenue\b',
                r'\barbitrat(e|ion)\b', r'\bdispute resolution\b', r'\bcourts of\b',
                r'\bapplicable law\b'
            ]
        },
        'Liability': {
            'title': 'Limitation of Liability',
            'badge_color': 'warning',
            'patterns': [
                r'\blimitation of liability\b', r'\bliab(le|ility)\b',
                r'\bindirect(,)? (incidental|consequential|punitive)\b',
                r'\baggregate liability\b', r'\bin no event shall (either|any) party\b',
                r'\bmaximum liability\b'
            ]
        },
        'Indemnification': {
            'title': 'Indemnification & Hold Harmless',
            'badge_color': 'secondary',
            'patterns': [
                r'\bindemnif(y|ied|ication|ies)\b', r'\bhold harmless\b', r'\bdefend and hold\b',
                r'\blosses(,)? damages(,)? and expenses\b'
            ]
        },
        'Payment': {
            'title': 'Payment & Consideration',
            'badge_color': 'success',
            'patterns': [
                r'\bpayment terms?\b', r'\binvoic(e|ing|es)\b', r'\bremuneration\b',
                r'\bcompensation\b', r'\bfees?\b', r'\bpayable within\b',
                r'\bnet \d+\b', r'\btaxes?\b', r'\bpurchase price\b'
            ]
        },
        'Intellectual Property': {
            'title': 'Intellectual Property Rights',
            'badge_color': 'info',
            'patterns': [
                r'\bintellectual property\b', r'\btrademark[s]?\b', r'\bcopyright[s]?\b',
                r'\bpatent[s]?\b', r'\bwork made for hire\b', r'\bproprietary rights\b',
                r'\blicense grant\b', r'\bownership of deliverables\b'
            ]
        },
        'Warranties': {
            'title': 'Warranties & Representations',
            'badge_color': 'warning',
            'patterns': [
                r'\brepresents and warrants\b', r'\bwarrant(y|ies|ed)?\b',
                r'\bas is\b', r'\bdisclaimer of warranties\b', r'\bmerchantability\b',
                r'\bfitness for a particular purpose\b'
            ]
        },
        'Force Majeure': {
            'title': 'Force Majeure',
            'badge_color': 'dark',
            'patterns': [
                r'\bforce majeure\b', r'\bacts? of god\b', r'\bnatural disaster[s]?\b',
                r'\bwar(,)? terrorism\b', r'\bbeyond (reasonable|the) control\b', r'\bepidemic|pandemic\b'
            ]
        },
        'Entire Agreement': {
            'title': 'Entire Agreement & Severability',
            'badge_color': 'secondary',
            'patterns': [
                r'\bentire agreement\b', r'\bseverab(le|ility)\b', r'\bsupersedes?\b',
                r'\binvalid or unenforceable\b', r'\bcomplete understanding\b',
                r'\bamendment[s]? in writing\b'
            ]
        }
    }

    # Date regex patterns (handles textual months, DD/MM/YYYY, YYYY-MM-DD, and relative periods)
    DATE_PATTERNS = [
        # Written months: "October 23, 2026", "23rd October 2026", "Oct. 23, 2026"
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[.]?\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b',
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[.]?\s+\d{4}\b',
        # ISO format: 2026-10-23
        r'\b\d{4}-\d{2}-\d{2}\b',
        # Standard slash/hyphen formats: 23/10/2026 or 10/23/2026
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        # Notice periods: "within 30 days", "within sixty (60) days", "12 months"
        r'\b(?:within\s+)?(?:\d+|one|two|three|four|five|ten|fifteen|twenty|thirty|sixty|ninety)\s*(?:\([0-9]+\)\s*)?(?:business\s+|calendar\s+)?(?:days|months|years|weeks)\b'
    ]

    # Monetary expressions: currency symbols ($ € £ ₹ ¥) and ISO codes (USD, EUR, GBP, INR, CAD)
    MONETARY_PATTERNS = [
        r'[\$€£₹¥]\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?(?:\s*(?:million|billion|thousand|crore|lakh))?',
        r'\b(?:USD|EUR|GBP|INR|CAD|AUD)\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?(?:\s*(?:million|billion|thousand|crore|lakh))?\b',
        r'\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:dollars|euros|pounds|rupees)\b'
    ]

    # Specific legal marker patterns
    OBLIGATION_REGEX = re.compile(
        r'\b(shall|must|is obligated to|are obligated to|agrees? to|undertakes? to|is required to|are required to)\b',
        re.IGNORECASE
    )
    CONDITION_REGEX = re.compile(
        r'\b(if|provided that|in the event that|in the event of|subject to|on condition that|contingent upon)\b',
        re.IGNORECASE
    )
    EXCEPTION_REGEX = re.compile(
        r'\b(except as|except|unless|notwithstanding|save for|excluding|provided however|without prejudice to)\b',
        re.IGNORECASE
    )

    def __init__(self):
        # Compile clause patterns for performance
        self._compiled_clauses = {}
        for category, info in self.CLAUSE_DEFINITIONS.items():
            self._compiled_clauses[category] = [
                re.compile(p, re.IGNORECASE) for p in info['patterns']
            ]
        self._compiled_dates = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]
        self._compiled_money = [re.compile(p, re.IGNORECASE) for p in self.MONETARY_PATTERNS]

    def extract(self, processed_sentences: List[Dict[str, Any]], raw_text: str) -> Dict[str, Any]:
        """
        Extract legal clauses, dates, amounts, obligations, conditions, and exceptions.
        
        Args:
            processed_sentences: List of sentence dictionaries from preprocessor.
            raw_text: Original raw extracted text string.
            
        Returns:
            Structured extraction dictionary.
        """
        clauses_by_category = {cat: [] for cat in self.CLAUSE_DEFINITIONS}
        obligations = []
        conditions = []
        exceptions = []

        # Analyze each sentence
        for s in processed_sentences:
            s_text = s['text']
            idx = s['index']

            # 1. Match Clauses
            for category, patterns in self._compiled_clauses.items():
                matched_keywords = []
                for pat in patterns:
                    match = pat.search(s_text)
                    if match:
                        matched_keywords.append(match.group(0))

                if matched_keywords:
                    clauses_by_category[category].append({
                        'sentence_index': idx,
                        'text': s_text,
                        'matched_keyword': matched_keywords[0]
                    })

            # 2. Match Obligations ("shall", "must")
            obl_match = self.OBLIGATION_REGEX.search(s_text)
            if obl_match:
                obligations.append({
                    'sentence_index': idx,
                    'text': s_text,
                    'keyword': obl_match.group(0).lower()
                })

            # 3. Match Conditions ("if", "provided that")
            cond_match = self.CONDITION_REGEX.search(s_text)
            if cond_match:
                conditions.append({
                    'sentence_index': idx,
                    'text': s_text,
                    'keyword': cond_match.group(0).lower()
                })

            # 4. Match Exceptions ("notwithstanding", "except", "unless")
            exc_match = self.EXCEPTION_REGEX.search(s_text)
            if exc_match:
                exceptions.append({
                    'sentence_index': idx,
                    'text': s_text,
                    'keyword': exc_match.group(0).lower()
                })

        # 5. Extract Dates & Monetary Amounts across the document
        extracted_dates = self._extract_dates(raw_text)
        extracted_monetary = self._extract_monetary(raw_text)

        # 6. Format clause results
        formatted_clauses = []
        for cat_key, matches in clauses_by_category.items():
            if matches:
                def_info = self.CLAUSE_DEFINITIONS[cat_key]
                formatted_clauses.append({
                    'category_id': cat_key,
                    'title': def_info['title'],
                    'badge_color': def_info['badge_color'],
                    'count': len(matches),
                    'matches': matches[:5]  # Cap at top 5 matches per clause for readability
                })

        # Sort clauses by occurrence count descending
        formatted_clauses.sort(key=lambda c: c['count'], reverse=True)

        return {
            'clauses': formatted_clauses,
            'dates': extracted_dates,
            'monetary_amounts': extracted_monetary,
            'obligations': obligations[:15],  # Cap for UI clarity
            'conditions': conditions[:15],
            'exceptions': exceptions[:15],
            'stats': {
                'total_clauses_identified': sum(c['count'] for c in formatted_clauses),
                'total_clause_categories': len(formatted_clauses),
                'total_dates': len(extracted_dates),
                'total_monetary': len(extracted_monetary),
                'total_obligations': len(obligations),
                'total_conditions': len(conditions),
                'total_exceptions': len(exceptions)
            }
        }

    def _extract_dates(self, text: str) -> List[Dict[str, str]]:
        """Find unique dates and their short textual contexts."""
        found = []
        seen = set()

        for pattern in self._compiled_dates:
            for match in pattern.finditer(text):
                date_str = match.group(0).strip()
                # Clean up punctuation around date
                date_clean = re.sub(r'^[,\.\s]+|[,\.\s]+$', '', date_str)
                if date_clean.lower() not in seen and len(date_clean) > 3:
                    seen.add(date_clean.lower())
                    start = max(0, match.start() - 40)
                    end = min(len(text), match.end() + 40)
                    context_snippet = text[start:end].replace('\n', ' ').strip()

                    found.append({
                        'date': date_clean,
                        'context': f"...{context_snippet}..."
                    })

        return found[:12]  # Cap at 12 representative dates

    def _extract_monetary(self, text: str) -> List[Dict[str, str]]:
        """Find unique monetary figures and their context."""
        found = []
        seen = set()

        for pattern in self._compiled_money:
            for match in pattern.finditer(text):
                money_str = match.group(0).strip()
                money_clean = re.sub(r'^[,\.\s]+|[,\.\s]+$', '', money_str)
                if money_clean.lower() not in seen:
                    seen.add(money_clean.lower())
                    start = max(0, match.start() - 40)
                    end = min(len(text), match.end() + 40)
                    context_snippet = text[start:end].replace('\n', ' ').strip()

                    found.append({
                        'amount': money_clean,
                        'context': f"...{context_snippet}..."
                    })

        return found[:12]
