"""
Legal Document Summarizer — Web Application Entry Point
College Mini Project in AI & Natural Language Processing
Built with Flask, PyMuPDF, python-docx, scikit-learn, and NetworkX.
"""

import os
import json
import uuid
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, Response, session
)
from werkzeug.utils import secure_filename

from config import Config
from modules.document_processor import DocumentProcessor, DocumentProcessingError
from modules.text_preprocessing import LegalTextPreprocessor
from modules.summarizer import LegalSummarizer
from modules.information_extractor import LegalInformationExtractor

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# Instantiate processing modules
doc_processor = DocumentProcessor(upload_folder=app.config['UPLOAD_FOLDER'])
text_preprocessor = LegalTextPreprocessor()
summarizer = LegalSummarizer(
    damping_factor=app.config['DAMPING_FACTOR'],
    max_iter=app.config['MAX_ITERATIONS']
)
info_extractor = LegalInformationExtractor()

# In-memory storage for recently processed sessions (for download route)
# Key: session_id, Value: results dictionary
RESULTS_CACHE = {}

# Sample legal contract templates for one-click instant demonstration in project viva
SAMPLE_DOCUMENTS = {
    "nda": {
        "title": "Non-Disclosure & Confidentiality Agreement (NDA)",
        "content": """MUTUAL NON-DISCLOSURE AND CONFIDENTIALITY AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of October 15, 2026 ("Effective Date"), by and between Apex Innovations Inc., a Delaware corporation with its principal place of business at 100 Tech Parkway, Suite 400, San Francisco, CA 94105 ("Disclosing Party"), and Horizon Ventures LLC, having its principal office at 250 Enterprise Blvd, New York, NY 10001 ("Receiving Party").

1. PURPOSE
The parties wish to explore a potential strategic business collaboration regarding autonomous cloud infrastructure. In connection with this purpose, each party may disclose to the other certain confidential and proprietary information.

2. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" refers to any non-public information, technical data, trade secrets, know-how, software code, financial forecasts, customer lists, and business strategies disclosed by one party to the other, whether orally, visually, or in tangible written form.

3. EXCLUSIONS FROM CONFIDENTIALITY
Confidential Information does not include information that:
(a) is or becomes publicly known through no breach of this Agreement by Receiving Party;
(b) was already in Receiving Party's rightful possession prior to disclosure without confidentiality obligations;
(c) is independently developed by Receiving Party without reference to or reliance upon Disclosing Party's Confidential Information; or
(d) is required to be disclosed pursuant to a valid judicial order or applicable statute, provided that Receiving Party gives prompt written notice to Disclosing Party.

4. OBLIGATIONS OF RECEIVING PARTY
Receiving Party shall hold and maintain the Confidential Information in strict confidence, exercising at least the same degree of care it uses to protect its own confidential materials of like importance, but in no event less than a reasonable degree of care. Receiving Party shall not disclose, reproduce, or distribute any Confidential Information to any third party without prior written consent of Disclosing Party. Receiving Party must restrict access to Confidential Information solely to those employees and legal advisors who have a verifiable need to know.

5. MONETARY CONSIDERATION AND PENALTY
In the event of an intentional or material breach of Section 4 by Receiving Party, Receiving Party agrees to pay liquidated damages in the amount of $250,000 to Disclosing Party within 30 days of written notice. Furthermore, Disclosing Party shall be entitled to seek injunctive relief in addition to monetary damages.

6. TERM AND TERMINATION
This Agreement shall remain in effect for a period of two (2) years from the Effective Date. Either party may terminate discussions at any time upon thirty (30) days prior written notice. Notwithstanding the termination of this Agreement, the confidentiality obligations with respect to trade secrets shall survive indefinitely, and with respect to all other Confidential Information shall survive for five (5) years following the date of disclosure.

7. GOVERNING LAW AND DISPUTE RESOLUTION
This Agreement shall be governed by and construed in accordance with the laws of the State of California, without regard to its conflict of law principles. Any dispute arising under or relating to this Agreement shall be submitted to binding arbitration in San Francisco, California under the Commercial Arbitration Rules of the American Arbitration Association.

8. ENTIRE AGREEMENT
This Agreement constitutes the entire understanding between the parties concerning the subject matter hereof and supersedes all prior discussions, representations, and agreements. No modification of this Agreement shall be binding unless executed in writing by authorized representatives of both parties."""
    },
    "service_contract": {
        "title": "Master Software Services Agreement (MSA)",
        "content": """MASTER SOFTWARE DEVELOPMENT AND SERVICES AGREEMENT

This Master Services Agreement ("Agreement") is made and entered into this 1st day of September, 2026, by and between CloudCore Technologies Ltd. ("Provider") and Global Retail Solutions Corp. ("Client").

1. SCOPE OF SERVICES
Provider shall design, develop, deploy, and maintain custom enterprise inventory management software as described in Statement of Work No. 1 ("SOW"). Provider warrants that all services shall be performed in a professional, workmanlike manner conforming to prevailing industry standards.

2. COMPENSATION AND PAYMENT TERMS
Client shall pay Provider a fixed fee of $120,000 for Milestone 1 deliverables. Invoices will be submitted on the first calendar day of each month. Client must remit payment within 30 calendar days following receipt of an undisputed invoice. Any late payments shall accrue interest at the rate of 1.5% per month or the maximum rate permitted by law.

3. INTELLECTUAL PROPERTY RIGHTS
Upon full payment of all fees due hereunder, Client shall own all right, title, and interest in and to the custom deliverables. Notwithstanding the foregoing, Provider retains all rights, title, and interest in its pre-existing libraries, algorithms, developer toolkits, and general know-how.

4. INDEMNIFICATION AND LIABILITY
Provider agrees to defend, indemnify, and hold harmless Client, its officers, and employees against any third-party claims alleging that the deliverables infringe any registered patent or copyright. IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL, OR PUNITIVE DAMAGES. Provider's total aggregate liability arising out of or related to this Agreement shall not exceed the total fees paid by Client under the applicable SOW during the preceding 12 months.

5. FORCE MAJEURE
Neither party shall be held responsible for delay or failure in performance resulting from acts of God, natural disasters, war, terrorism, government sanction, or nationwide power grid failures beyond reasonable control, provided that the affected party provides prompt written notice within seven (7) days of occurrence.

6. TERMINATION FOR CAUSE
Either party may terminate this Agreement immediately if the other party commits a material breach and fails to cure such breach within forty-five (45) days after receiving written notice.

7. GOVERNING LAW
This Agreement shall be governed by and interpreted pursuant to the laws of the State of New York. The parties consent to the exclusive jurisdiction of the state and federal courts located in New York County, New York."""
    }
}


@app.route('/', methods=['GET'])
def index():
    """Render home page with upload form and sample legal contracts."""
    return render_template(
        'index.html',
        sample_documents=SAMPLE_DOCUMENTS,
        allowed_extensions=list(Config.ALLOWED_EXTENSIONS)
    )


@app.route('/process', methods=['POST'])
def process_document():
    """
    Handle document upload or sample text submission, run NLP pipeline,
    and display interactive analysis dashboard.
    """
    try:
        raw_text = ""
        filename = ""
        source_type = request.form.get('source_type', 'upload')
        summarization_method = request.form.get('summarization_method', 'textrank')

        if summarization_method not in ['textrank', 'tfidf']:
            summarization_method = 'textrank'

        # Case 1: Pre-loaded Sample Document selected
        if source_type == 'sample':
            sample_key = request.form.get('sample_key')
            if sample_key not in SAMPLE_DOCUMENTS:
                flash("Selected sample contract not found.", "warning")
                return redirect(url_for('index'))

            sample_data = SAMPLE_DOCUMENTS[sample_key]
            raw_text = sample_data['content']
            filename = f"{sample_key}_sample.txt"
            metadata = {
                'filename': filename,
                'file_extension': 'txt',
                'file_size_formatted': f"{len(raw_text.encode('utf-8')) / 1024:.1f} KB",
                'char_count': len(raw_text),
                'word_count': len(raw_text.split()),
                'line_count': len(raw_text.splitlines())
            }

        # Case 2: Direct Text Input
        elif source_type == 'text':
            raw_text = request.form.get('pasted_text', '').strip()
            if not raw_text:
                flash("Please paste or enter legal document text to proceed.", "danger")
                return redirect(url_for('index'))

            filename = "Pasted_Legal_Document.txt"
            metadata = {
                'filename': filename,
                'file_extension': 'txt',
                'file_size_formatted': f"{len(raw_text.encode('utf-8')) / 1024:.1f} KB",
                'char_count': len(raw_text),
                'word_count': len(raw_text.split()),
                'line_count': len(raw_text.splitlines())
            }

        # Case 3: File Upload (.pdf, .docx, .txt)
        else:
            if 'file' not in request.files:
                flash("No file was submitted in the request.", "danger")
                return redirect(url_for('index'))

            file = request.files['file']
            if not file or file.filename == '':
                flash("No file was selected. Please choose a PDF, DOCX, or TXT file.", "warning")
                return redirect(url_for('index'))

            if not DocumentProcessor.allowed_file(file.filename):
                flash(f"Invalid file extension. Allowed file types are: {', '.join(Config.ALLOWED_EXTENSIONS).upper()}", "danger")
                return redirect(url_for('index'))

            # Secure filename and save to uploads folder
            orig_filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex[:8]}_{orig_filename}"
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(saved_path)

            try:
                # Extract text using document processor
                raw_text, metadata = doc_processor.extract_text(saved_path)
                filename = orig_filename
            finally:
                # Clean up uploaded file from disk after reading into memory
                if os.path.exists(saved_path):
                    try:
                        os.remove(saved_path)
                    except OSError:
                        pass

        # Step 1: Preprocess Text with Legal Preprocessor
        processed_sentences = text_preprocessor.process(raw_text)

        if not processed_sentences:
            flash("The provided document does not contain recognizable sentences for summarization.", "warning")
            return redirect(url_for('index'))

        # Step 2: Extractive Summarization (Short, Medium, Detailed)
        summaries = summarizer.summarize(processed_sentences, method=summarization_method)

        # Step 3: Legal Information & Clause Extraction
        extracted_info = info_extractor.extract(processed_sentences, raw_text)

        # Compute Document Reading Statistics
        total_words = metadata.get('word_count', len(raw_text.split()))
        reading_time_mins = max(1, round(total_words / 200))  # Standard 200 wpm reading speed

        # Package full result
        session_id = uuid.uuid4().hex
        result_payload = {
            'session_id': session_id,
            'filename': filename,
            'processed_at': datetime.now().strftime("%B %d, %Y - %I:%M %p"),
            'metadata': metadata,
            'total_sentences': len(processed_sentences),
            'reading_time_mins': reading_time_mins,
            'summaries': summaries,
            'extracted_info': extracted_info,
            'all_sentences': processed_sentences,
            'raw_text': raw_text
        }

        # Cache result in memory for download endpoints
        RESULTS_CACHE[session_id] = result_payload

        # Keep cache bounded (limit to 20 most recent sessions)
        if len(RESULTS_CACHE) > 20:
            oldest_key = next(iter(RESULTS_CACHE))
            del RESULTS_CACHE[oldest_key]

        return render_template('results.html', data=result_payload)

    except DocumentProcessingError as e:
        flash(f"Document Extraction Error: {str(e)}", "danger")
        return redirect(url_for('index'))
    except Exception as e:
        flash(f"An unexpected processing error occurred: {str(e)}", "danger")
        return redirect(url_for('index'))


@app.route('/download/<session_id>/<level>/<fmt>', methods=['GET'])
def download_summary(session_id: str, level: str, fmt: str):
    """
    Download the generated summary as a formatted TXT or JSON document.
    """
    if session_id not in RESULTS_CACHE:
        flash("Summary session expired or not found. Please re-upload your document.", "warning")
        return redirect(url_for('index'))

    data = RESULTS_CACHE[session_id]
    if level not in ['short', 'medium', 'detailed']:
        level = 'medium'

    summary_obj = data['summaries'].get(level, {})
    filename_base = data['filename'].rsplit('.', 1)[0]

    if fmt == 'json':
        export_data = {
            "title": f"Legal Document Summary - {data['filename']}",
            "generated_on": data['processed_at'],
            "summary_level": level,
            "method": data['summaries'].get('method_used', 'textrank'),
            "reduction_percentage": f"{summary_obj.get('reduction_percent', 0)}%",
            "original_word_count": data['metadata'].get('word_count', 0),
            "summary_word_count": summary_obj.get('word_count', 0),
            "summary_text": summary_obj.get('text', ''),
            "extracted_clauses": data['extracted_info']['clauses'],
            "dates_found": data['extracted_info']['dates'],
            "monetary_amounts": data['extracted_info']['monetary_amounts'],
            "notice": "This document provides preliminary extractive assistance and does not constitute formal legal advice."
        }
        json_output = json.dumps(export_data, indent=2)
        return Response(
            json_output,
            mimetype="application/json",
            headers={"Content-disposition": f"attachment; filename={filename_base}_{level}_summary.json"}
        )

    else:  # Text format default
        lines = [
            "=" * 70,
            "LEGAL DOCUMENT EXTRACTIVE SUMMARY REPORT",
            "=" * 70,
            f"Document Name    : {data['filename']}",
            f"Generated At     : {data['processed_at']}",
            f"Summary Presets  : {level.upper()}",
            f"Algorithm Used   : {data['summaries'].get('method_used', 'TextRank').upper()}",
            f"Original Words   : {data['metadata'].get('word_count', 0)}",
            f"Summary Words    : {summary_obj.get('word_count', 0)}",
            f"Compression Ratio: {summary_obj.get('reduction_percent', 0)}% text reduction",
            "=" * 70,
            "",
            "--- SUMMARY TEXT ---",
            summary_obj.get('text', ''),
            "",
            "=" * 70,
            "--- KEY LEGAL CLAUSES IDENTIFIED ---"
        ]

        for c in data['extracted_info']['clauses']:
            lines.append(f"\n[ {c['title']} ({c['count']} provisions found) ]")
            for m in c['matches']:
                lines.append(f"  * {m['text']}")

        lines.extend([
            "",
            "=" * 70,
            "NOTICE & DISCLAIMER:",
            "This summary is generated using Natural Language Processing (Extractive Summarization)",
            "for preliminary document assistance and study purposes only.",
            "It does NOT constitute formal legal advice.",
            "=" * 70
        ])

        txt_output = "\n".join(lines)
        return Response(
            txt_output,
            mimetype="text/plain",
            headers={"Content-disposition": f"attachment; filename={filename_base}_{level}_summary.txt"}
        )


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle uploads exceeding max content length."""
    flash("The uploaded file exceeds the 16MB file size limit. Please upload a smaller document.", "danger")
    return redirect(url_for('index')), 413


@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 errors."""
    return render_template('index.html', error="Page not found."), 404


if __name__ == '__main__':
    # Run development server
    print("=" * 65)
    print(" Legal Document Summarizer — Web Application Running")
    print(" Access URL: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host='127.0.0.1', port=5000, debug=True)
