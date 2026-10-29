"""
Text Preprocessing Module
Specialized NLP preprocessor for legal documents.
Handles legal abbreviations, sentence boundary disambiguation, legal stopword handling,
negation preservation, tokenization, and sentence metadata extraction.
"""

import re
from typing import List, Dict, Any, Tuple


class LegalTextPreprocessor:
    """Preprocesses raw legal text into structured, traceable sentences and clean tokens."""

    # Common legal and standard English abbreviations that end with a period
    # These must NOT trigger false sentence boundaries.
    ABBREVIATIONS = [
        r'\bv\.', r'\bvs\.', r'\bno\.', r'\bnos\.', r'\bsec\.', r'\bsecs\.',
        r'\bart\.', r'\barts\.', r'\bcl\.', r'\bpara\.', r'\bparas\.',
        r'\binc\.', r'\bltd\.', r'\bcorp\.', r'\bco\.', r'\bllc\.', r'\bllp\.',
        r'\bu\.s\.', r'\bu\.k\.', r'\be\.g\.', r'\bi\.e\.', r'\bet al\.',
        r'\bcf\.', r'\bviz\.', r'\bp\.a\.', r'\bjan\.', r'\bfeb\.', r'\bmar\.',
        r'\bapr\.', r'\bjul\.', r'\baug\.', r'\bsep\.', r'\bsept\.', r'\boct\.',
        r'\bnov\.', r'\bdec\.', r'\bmr\.', r'\bms\.', r'\bmrs\.', r'\bdr\.',
        r'\bprof\.', r'\bhon\.', r'\bsr\.', r'\bjr\.', r'\bex\.', r'\bre\.'
    ]

    # Legal stop words: general filler words that do not carry topic significance.
    # CRITICAL: We deliberately EXCLUDE negations ('not', 'no', 'never', 'neither', 'nor', 'without')
    # and exception markers ('except', 'unless') to maintain legal truthfulness.
    STOP_WORDS = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
        'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
        'below', 'between', 'both', 'by', 'can', 'cannot', 'could', 'couldn\'t', 'did',
        'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each',
        'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have',
        'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s',
        'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll',
        'i\'m', 'i\'ve', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself',
        'let\'s', 'me', 'more', 'most', 'my', 'myself', 'of', 'off', 'on', 'once', 'only',
        'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
        'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such',
        'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
        'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve',
        'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
        'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what',
        'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s',
        'whom', 'why', 'why\'s', 'with', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll',
        'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves',
        # Standard legalese filler words that dilute TF-IDF:
        'herein', 'therein', 'wherein', 'hereof', 'thereof', 'whereof', 'hereunder',
        'thereunder', 'hereinafter', 'hereto', 'thereto', 'hereby', 'thereby'
    }

    # Negation and qualification terms critical for legal veracity
    NEGATION_TERMS = {
        'not', 'no', 'never', 'neither', 'nor', 'none', 'nothing', 'nobody', 'nowhere',
        'without', 'prohibited', 'forbidden', 'disclaim', 'disclaims', 'disclaimed',
        'void', 'invalid', 'inadmissible', 'waive', 'waives', 'waived', 'unauthorized'
    }

    # Obligation markers (crucial for legal ranking)
    OBLIGATION_PATTERNS = [
        r'\bshall\b', r'\bmust\b', r'\bagrees? to\b', r'\bundertakes? to\b',
        r'\bis obligated to\b', r'\bare obligated to\b', r'\bcovenants?\b',
        r'\bis required to\b', r'\bare required to\b', r'\bwarrants?\b'
    ]

    # Condition markers
    CONDITION_PATTERNS = [
        r'\bif\b', r'\bprovided that\b', r'\bin the event that\b', r'\bin the event of\b',
        r'\bsubject to\b', r'\bon condition that\b', r'\bcontingent upon\b', r'\bwhere\b'
    ]

    # Exception markers
    EXCEPTION_PATTERNS = [
        r'\bexcept\b', r'\bexcept as\b', r'\bunless\b', r'\bnotwithstanding\b',
        r'\bsave for\b', r'\bexcluding\b', r'\bprovided however\b', r'\bwithout prejudice to\b'
    ]

    def __init__(self):
        # Compile abbreviation replacement regexes
        self._abbr_patterns = [(re.compile(abbr, re.IGNORECASE), abbr) for abbr in self.ABBREVIATIONS]

    def normalize_text(self, text: str) -> str:
        """
        Normalize typographic oddities, whitespace, and smart punctuation.
        """
        if not text:
            return ""

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Replace smart quotes and dashes with ASCII equivalents
        replacements = {
            '“': '"', '”': '"', '‘': "'", '’': "'",
            '—': ' - ', '–': ' - ', '…': '...', '\t': ' '
        }
        for orig, rep in replacements.items():
            text = text.replace(orig, rep)

        # Fix hyphenated line breaks (e.g., "agree-\nment" -> "agreement")
        text = re.sub(r'(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)', r'\1\2', text)

        # Merge soft line breaks within paragraphs, preserving double newlines
        paragraphs = text.split('\n\n')
        normalized_paragraphs = []
        for p in paragraphs:
            # Replace single newlines within a paragraph with spaces
            cleaned_p = re.sub(r'(?<!\n)\n(?!\n)', ' ', p)
            # Collapse multiple spaces
            cleaned_p = re.sub(r'[ ]{2,}', ' ', cleaned_p).strip()
            if cleaned_p:
                normalized_paragraphs.append(cleaned_p)

        return '\n\n'.join(normalized_paragraphs)

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split legal text into clean sentences while protecting abbreviations,
        citations, numbered lists, and decimal numbers.
        """
        text = self.normalize_text(text)
        if not text:
            return []

        # Step 1: Protect legal abbreviations by replacing their dots with a placeholder
        protected_text = text
        placeholder = "§DOT§"

        # Protect decimal numbers (e.g., 1.5, 3.14, Section 2.4)
        protected_text = re.sub(r'(\d+)\.(\d+)', r'\1' + placeholder + r'\2', protected_text)

        # Protect known abbreviations
        for pattern, _ in self._abbr_patterns:
            matches = list(pattern.finditer(protected_text))
            for match in reversed(matches):
                start, end = match.span()
                matched_str = protected_text[start:end]
                protected_str = matched_str.replace('.', placeholder)
                protected_text = protected_text[:start] + protected_str + protected_text[end:]

        # Step 2: Split sentences based on punctuation followed by whitespace and capital letter or numbering
        # or followed by paragraph boundaries
        sentence_delimiters = re.compile(
            r'(?<=[.?!;])\s+(?=[A-Z0-9\(\"\'\[])|(?<=\n\n)|(?<=[.?!])\n+'
        )
        raw_chunks = sentence_delimiters.split(protected_text)

        # Step 3: Restore placeholders and clean sentences
        sentences = []
        for chunk in raw_chunks:
            # Restore protected dots
            restored = chunk.replace(placeholder, '.').strip()
            # Clean up residual whitespace
            restored = re.sub(r'\s+', ' ', restored).strip()

            # Filter out very short or empty strings (e.g., stray punctuation or bullet markers)
            if len(restored) >= 5 and re.search(r'[a-zA-Z0-9]', restored):
                # If chunk is an orphaned bullet/number (e.g. "1." or "(a)") attach or handle gracefully
                sentences.append(restored)

        # Step 4: Merge any dangling or fragmented sentences
        merged_sentences = []
        for s in sentences:
            if not merged_sentences:
                merged_sentences.append(s)
            else:
                last = merged_sentences[-1]
                # If current sentence starts with lowercase and previous didn't end with strong stop, merge
                if s[0].islower() and not last.endswith(('.', '!', '?')):
                    merged_sentences[-1] = f"{last} {s}"
                # If previous was very short header like "Section 1." or "ARTICLE III.", merge
                elif re.match(r'^(Section|Article|Clause|\d+\.|\([a-z0-9]+\))\s*$', last, re.IGNORECASE):
                    merged_sentences[-1] = f"{last} {s}"
                else:
                    merged_sentences.append(s)

        return merged_sentences

    def tokenize_sentence(self, sentence: str) -> List[str]:
        """
        Tokenize a sentence into cleaned, lowercase alphanumeric tokens
        excluding non-negation stop words.
        """
        # Convert to lowercase
        text_lower = sentence.lower()
        # Find alphanumeric words (including hyphenated terms)
        words = re.findall(r'\b[a-z0-9][a-z0-9_-]*\b', text_lower)
        # Filter stopwords, but explicitly keep negations
        filtered = [
            w for w in words
            if (w not in self.STOP_WORDS or w in self.NEGATION_TERMS) and len(w) > 1
        ]
        return filtered

    def process(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Full preprocessing pipeline.
        
        Returns:
            List of sentence dictionaries containing:
            - index: int (0-based)
            - text: str (original sentence text)
            - clean_text: str (space-joined filtered tokens)
            - tokens: List[str]
            - word_count: int
            - has_negation: bool
            - has_obligation: bool
            - has_condition: bool
            - has_exception: bool
        """
        raw_sentences = self.split_into_sentences(raw_text)
        processed = []

        for idx, s in enumerate(raw_sentences):
            tokens = self.tokenize_sentence(s)
            clean_text = " ".join(tokens)
            words = s.split()

            # Detect legal linguistic attributes
            has_negation = any(neg in tokens for neg in self.NEGATION_TERMS)
            has_obligation = any(re.search(p, s, re.IGNORECASE) for p in self.OBLIGATION_PATTERNS)
            has_condition = any(re.search(p, s, re.IGNORECASE) for p in self.CONDITION_PATTERNS)
            has_exception = any(re.search(p, s, re.IGNORECASE) for p in self.EXCEPTION_PATTERNS)

            processed.append({
                'index': idx,
                'text': s,
                'clean_text': clean_text,
                'tokens': tokens,
                'word_count': len(words),
                'char_count': len(s),
                'has_negation': has_negation,
                'has_obligation': has_obligation,
                'has_condition': has_condition,
                'has_exception': has_exception
            })

        return processed
