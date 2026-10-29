"""
Legal Summarizer Module
Implements Extractive Summarization using:
1. TF-IDF + Cosine Similarity (Centroid Salience Model)
2. TextRank (Graph-based PageRank ranking)
Includes legal salience weighting (boosting obligations, exceptions, and key legal provisions).
Guarantees 100% extractive fidelity to prevent legal fact hallucination.
"""

import math
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


class LegalSummarizer:
    """
    Extractive summarization engine designed specifically for contracts,
    statutes, agreements, and legal briefs.
    """

    # Multipliers to boost sentences containing critical legal signposts
    BOOST_OBLIGATION = 1.25    # "shall", "must", "agrees to"
    BOOST_EXCEPTION = 1.20     # "except", "notwithstanding", "unless"
    BOOST_CONDITION = 1.15     # "if", "provided that", "in the event"
    BOOST_EARLY_POSITION = 1.10 # Preamble, party identification, effective dates

    def __init__(self, damping_factor: float = 0.85, max_iter: int = 100):
        self.damping_factor = damping_factor
        self.max_iter = max_iter

    def summarize(
        self,
        processed_sentences: List[Dict[str, Any]],
        method: str = "textrank"
    ) -> Dict[str, Any]:
        """
        Generate Short, Medium, and Detailed extractive summaries.
        
        Args:
            processed_sentences: List of preprocessed sentence dicts from LegalTextPreprocessor.
            method: 'textrank' or 'tfidf' (default: 'textrank').
            
        Returns:
            Dictionary containing summaries for 'short', 'medium', 'detailed',
            along with scoring details and compression statistics.
        """
        total_sentences = len(processed_sentences)
        if total_sentences == 0:
            return {
                'short': {'text': '', 'sentence_indices': [], 'word_count': 0, 'reduction_percent': 0.0},
                'medium': {'text': '', 'sentence_indices': [], 'word_count': 0, 'reduction_percent': 0.0},
                'detailed': {'text': '', 'sentence_indices': [], 'word_count': 0, 'reduction_percent': 0.0},
                'method_used': method,
                'total_sentences': 0,
                'total_words': 0
            }

        total_words = sum(s['word_count'] for s in processed_sentences)

        # If document is very brief (<= 3 sentences), return full text for all levels
        if total_sentences <= 3:
            full_text = " ".join(s['text'] for s in processed_sentences)
            summary_payload = {
                'text': full_text,
                'sentence_indices': list(range(total_sentences)),
                'sentences': [s['text'] for s in processed_sentences],
                'sentence_count': total_sentences,
                'word_count': total_words,
                'reduction_percent': 0.0
            }
            return {
                'short': summary_payload,
                'medium': summary_payload,
                'detailed': summary_payload,
                'method_used': method,
                'total_sentences': total_sentences,
                'total_words': total_words
            }

        # Step 1: Compute base mathematical scores using selected algorithm
        if method == "tfidf":
            base_scores = self._compute_tfidf_scores(processed_sentences)
        else:
            base_scores = self._compute_textrank_scores(processed_sentences)

        # Step 2: Apply legal domain salience boosting
        final_scores = self._apply_legal_boosts(base_scores, processed_sentences)

        # Step 3: Extract summaries for Short, Medium, Detailed presets
        summaries = {}
        for level in ['short', 'medium', 'detailed']:
            count = self._get_sentence_count(total_sentences, level)
            summary_data = self._build_summary(processed_sentences, final_scores, count, total_words)
            summaries[level] = summary_data

        summaries['method_used'] = method
        summaries['total_sentences'] = total_sentences
        summaries['total_words'] = total_words

        return summaries

    def _compute_tfidf_scores(self, sentences: List[Dict[str, Any]]) -> np.ndarray:
        """
        Computes salience based on Cosine Similarity to Document Centroid Vector.
        Sentences whose TF-IDF vector is closest to the document's mean vector
        capture the most comprehensive topics of the agreement.
        """
        corpus = [s['clean_text'] for s in sentences]
        # If corpus has empty tokens, fallback to raw text
        if not any(c.strip() for c in corpus):
            corpus = [s['text'].lower() for s in sentences]

        try:
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Document Centroid Vector = column mean of all sentence TF-IDF vectors
            centroid_vector = np.asarray(tfidf_matrix.mean(axis=0))

            # Cosine similarity of each sentence to the centroid
            scores = cosine_similarity(tfidf_matrix, centroid_vector).flatten()
            
            # Normalize to 0..1 range
            max_s = np.max(scores) if np.max(scores) > 0 else 1.0
            return scores / max_s
        except Exception:
            # Fallback uniform scores
            return np.ones(len(sentences)) / len(sentences)

    def _compute_textrank_scores(self, sentences: List[Dict[str, Any]]) -> np.ndarray:
        """
        Computes sentence importance using the TextRank graph algorithm.
        Vertices represent sentences; edge weights represent TF-IDF cosine similarity.
        Sentences recommended by other central sentences receive high PageRank scores.
        """
        n = len(sentences)
        corpus = [s['clean_text'] for s in sentences]
        if not any(c.strip() for c in corpus):
            corpus = [s['text'].lower() for s in sentences]

        try:
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(corpus)
            sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

            # Build NetworkX weighted graph
            graph = nx.Graph()
            for i in range(n):
                graph.add_node(i)

            for i in range(n):
                for j in range(i + 1, n):
                    weight = float(sim_matrix[i][j])
                    if weight > 0.05:  # Similarity threshold to avoid noisy fully-connected graph
                        graph.add_edge(i, j, weight=weight)

            # Compute PageRank
            try:
                pagerank_scores = nx.pagerank(
                    graph,
                    alpha=self.damping_factor,
                    max_iter=self.max_iter,
                    weight='weight'
                )
                scores = np.array([pagerank_scores.get(i, 0.0) for i in range(n)])
            except Exception:
                # If power iteration fails, sum similarity row
                scores = np.sum(sim_matrix, axis=1)

            max_s = np.max(scores) if np.max(scores) > 0 else 1.0
            return scores / max_s

        except Exception:
            return np.ones(n) / n

    def _apply_legal_boosts(
        self,
        base_scores: np.ndarray,
        sentences: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Apply legal heuristics to refine extractive scores:
        - Obligations ('shall', 'must') reflect binding legal duties.
        - Exceptions ('except', 'unless') contain crucial carve-outs.
        - Conditions ('if', 'provided that') define trigger points.
        - Early sentences (first 2-3) often define the agreement parties & purpose.
        """
        boosted = np.copy(base_scores)
        total = len(sentences)

        for i, s in enumerate(sentences):
            multiplier = 1.0

            # Legal obligations
            if s.get('has_obligation', False):
                multiplier *= self.BOOST_OBLIGATION

            # Legal exceptions & negations
            if s.get('has_exception', False) or s.get('has_negation', False):
                multiplier *= self.BOOST_EXCEPTION

            # Legal conditions
            if s.get('has_condition', False):
                multiplier *= self.BOOST_CONDITION

            # Preamble bonus for first 2 sentences
            if i < 2:
                multiplier *= self.BOOST_EARLY_POSITION

            # Slight penalty for extremely short fragments (< 6 words) or overly long run-ons (> 70 words)
            w_count = s.get('word_count', 0)
            if w_count < 6:
                multiplier *= 0.60
            elif w_count > 70:
                multiplier *= 0.85

            boosted[i] *= multiplier

        return boosted

    @staticmethod
    def _get_sentence_count(total_sentences: int, level: str) -> int:
        """Calculate target sentence count according to preset level."""
        if total_sentences <= 4:
            return total_sentences

        if level == 'short':
            # ~15% or at least 3, at most 6 sentences
            target = max(3, min(int(math.ceil(total_sentences * 0.15)), 6))
        elif level == 'medium':
            # ~30% or at least 5, at most 12 sentences
            target = max(5, min(int(math.ceil(total_sentences * 0.30)), 12))
        elif level == 'detailed':
            # ~50% or at least 8, at most 25 sentences
            target = max(8, min(int(math.ceil(total_sentences * 0.50)), 25))
        else:
            target = max(3, int(total_sentences * 0.25))

        return min(target, total_sentences)

    @staticmethod
    def _build_summary(
        sentences: List[Dict[str, Any]],
        scores: np.ndarray,
        target_count: int,
        total_words: int
    ) -> Dict[str, Any]:
        """
        Select top-ranking sentences, reorder them chronologically (sentence index order)
        to maintain narrative coherence, and compute summary reduction metrics.
        """
        # Get indices of top scores
        top_indices = np.argsort(scores)[-target_count:]
        
        # Sort chronologically by original document sentence index
        chronological_indices = sorted(top_indices.tolist())

        selected_sentences = [sentences[idx]['text'] for idx in chronological_indices]
        summary_text = " ".join(selected_sentences)
        summary_words = sum(sentences[idx]['word_count'] for idx in chronological_indices)

        # Calculate reduction percentage
        if total_words > 0:
            reduction = max(0.0, round(((total_words - summary_words) / total_words) * 100, 1))
        else:
            reduction = 0.0

        return {
            'text': summary_text,
            'sentence_indices': chronological_indices,
            'sentences': selected_sentences,
            'sentence_count': len(chronological_indices),
            'word_count': summary_words,
            'reduction_percent': reduction
        }
