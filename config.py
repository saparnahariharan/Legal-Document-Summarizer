"""
Configuration settings for the Legal Document Summarizer web application.
This module defines application-wide parameters, file paths, upload restrictions,
and NLP summarization settings.
"""

import os
from pathlib import Path

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Base application configuration."""
    # Secret key for session management and flash messages
    SECRET_KEY = os.environ.get('SECRET_KEY', 'legal-doc-summarizer-secret-key-2026')

    # Uploads and Outputs directories
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')

    # File upload restrictions
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 Megabytes max upload limit

    # Summarization presets (ratio of sentences or sentence counts)
    SUMMARY_RATIOS = {
        'short': 0.15,     # ~15% of original sentences (min 3 sentences)
        'medium': 0.30,    # ~30% of original sentences (min 5 sentences)
        'detailed': 0.50   # ~50% of original sentences (min 8 sentences)
    }

    # Minimum and maximum sentence bounds for summary presets
    SUMMARY_BOUNDS = {
        'short': {'min': 3, 'max': 6},
        'medium': {'min': 6, 'max': 14},
        'detailed': {'min': 12, 'max': 30}
    }

    # TextRank / TF-IDF settings
    DAMPING_FACTOR = 0.85
    MAX_ITERATIONS = 100
    CONVERGENCE_THRESHOLD = 1e-4

    # Ensure required folders exist
    @staticmethod
    def init_app(app=None):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
