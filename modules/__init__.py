"""
Legal Document Summarizer Modules
Contains submodules for document text extraction, NLP preprocessing,
extractive summarization, and legal information extraction.
"""

from .document_processor import DocumentProcessor
from .text_preprocessing import LegalTextPreprocessor
from .summarizer import LegalSummarizer
from .information_extractor import LegalInformationExtractor

__all__ = [
    'DocumentProcessor',
    'LegalTextPreprocessor',
    'LegalSummarizer',
    'LegalInformationExtractor'
]
