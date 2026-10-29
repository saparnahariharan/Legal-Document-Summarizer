# Legal Document Summarizer — AI and NLP Based Web Application

An extractive Natural Language Processing (NLP) system and interactive web application designed to analyze, summarize, and extract critical intelligence from contracts, agreements, statutes, and legal briefs.

Built with **Python Flask**, **PyMuPDF**, **python-docx**, **scikit-learn**, and **NetworkX**.

---

## ⚖️ Legal Disclaimer Notice
> **Notice:** This application provides preliminary document assistance and automated extractive analysis using NLP for educational, academic, and research purposes. It **does not** constitute formal legal advice.

---

## 🌟 Key Features

1. **Multi-Format Document Ingestion**:
   - **PDF Documents (`.pdf`)**: High-speed, accurate text parsing using `PyMuPDF` with automatic fallback to `pypdf`. Handles encrypted and scanned document checks.
   - **Word Documents (`.docx`)**: Comprehensive extraction of paragraphs and table cells (schedules, compensation tables) using `python-docx`.
   - **Plain Text (`.txt`)**: Multi-encoding decoder (UTF-8, UTF-8-BOM, Windows-1252, Latin-1).

2. **Legal Domain NLP Preprocessing**:
   - **Legal Sentence Boundary Disambiguation**: Custom tokenizer protects legal abbreviations (`v.`, `Sec.`, `Art.`, `Inc.`, `Ltd.`, `U.S.`, `cl.`, `para.`, `et al.`) from false sentence splits.
   - **Preservation of Negations & Conditions**: Unlike standard stopword removers, words like `not`, `never`, `without`, `except`, `unless` are strictly preserved to maintain legal veracity.

3. **Extractive Summarization (Zero Hallucination)**:
   - **TextRank Algorithm**: Graph-based PageRank model scoring sentence connectivity.
   - **TF-IDF Centroid Similarity**: Vector-space model measuring sentence closeness to document thematic mean.
   - **Legal Salience Boosting**: Weights sentences containing mandatory duties (`shall`, `must`), exceptions (`notwithstanding`, `except as`), and core legal clauses.
   - **Presets**: Short (~15%), Medium (~30%), and Detailed (~50%) extractive summaries sorted chronologically.

4. **Legal Intelligence & Clause Extraction**:
   - **Clause Classification**: Identifies Termination, Confidentiality, Governing Law, Liabilities, Warranties, Payment Terms, and Force Majeure.
   - **Entity Detection**: Extracts Dates/Deadlines, Notice Periods, Monetary Values (`$`, `€`, `£`, `₹`, `USD`).
   - **Syntactic Markers**: Segregates binding **Obligations**, **Conditions**, and **Exceptions**.

5. **Interactive Results Dashboard**:
   - Multi-tab summary inspector with word count and compression percentage.
   - One-click **Copy to Clipboard** with visual toast feedback.
   - Export reports as formatted **Plain Text (`.txt`)** or structured **JSON (`.json`)**.
   - Built-in **Original Document Sentence Verifier** highlighting summary sentences in the source text.
   - Pre-loaded sample contracts (NDA, Master Services Agreement) for instant demonstration.

---

## 📁 Project Architecture

```
Legal-Document-Summarizer/
├── app.py                            # Flask application routes and controller
├── config.py                         # Application configuration and presets
├── requirements.txt                  # Python dependencies
├── README.md                         # Documentation and viva guide
├── .gitignore                        # Git exclusion rules
├── modules/
│   ├── __init__.py                   # Package exports
│   ├── document_processor.py         # PDF, DOCX, TXT extraction & validation
│   ├── text_preprocessing.py         # Legal NLP sentence segmentation & cleaning
│   ├── summarizer.py                 # Extractive algorithms (TextRank & TF-IDF)
│   └── information_extractor.py      # Clause categorization & entity mining
├── templates/
│   ├── index.html                    # Document upload & input interface
│   └── results.html                  # Professional analysis dashboard
├── static/
│   ├── css/
│   │   └── style.css                 # Custom legal-tech styling
│   └── js/
│       └── script.js                 # Drag-and-drop & clipboard interactions
└── tests/
    ├── test_document_processor.py    # Document parsing unit tests
    ├── test_summarizer.py            # NLP & summarization unit tests
    └── test_information_extractor.py # Clause & entity extraction unit tests
```

---

## 🚀 Quick Start (Windows)

### 1. Open Terminal in Project Folder
Open PowerShell or Command Prompt:
```powershell
cd "C:\Users\User\Desktop\Legal-Document-Summarizer"
```

### 2. (Optional) Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Run Unit Tests
Verify all document extraction, NLP, and summarization tests pass:
```powershell
pytest -v
```

### 5. Launch the Web Application
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🎓 College Mini Project Viva & Concept Guide

### 1. Why Extractive Summarization instead of Abstractive (LLM) for Legal Documents?
- **Abstractive Summarization** (e.g., GPT, T5) paraphrases text. In law, changing a single word (such as changing *"Party A may terminate"* to *"Party A shall terminate"*) completely alters legal rights and can result in severe liability or hallucination.
- **Extractive Summarization** selects the most salient, verbatim sentences directly from the original document. It guarantees **100% legal traceability** and eliminates generative hallucinations.

### 2. What is TF-IDF?
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$$
- **Term Frequency (TF)**: How often word $t$ appears in sentence $d$.
- **Inverse Document Frequency (IDF)**: Penalizes words that appear everywhere, highlighting words specific to that clause.
- **Centroid Cosine Similarity**: We average the TF-IDF vectors of all sentences to form a **Thematic Centroid** ($\vec{C}$). Sentences whose vectors have the highest cosine similarity $\frac{\vec{v}_i \cdot \vec{C}}{\|\vec{v}_i\|\|\vec{C}\|}$ represent the core topics of the contract.

### 3. What is TextRank?
TextRank models the document as an undirected weighted graph $G = (V, E)$:
- **Vertices ($V$)**: Sentences in the contract.
- **Edges ($E$)**: Semantic similarity between pairs of sentences (measured by TF-IDF vector overlap).
- **PageRank Algorithm**: Sentences vote on the importance of other sentences. Highly connected sentences receive high PageRank centrality:
$$PR(v_i) = \frac{1 - d}{|V|} + d \sum_{v_j \in In(v_i)} \frac{W_{ji}}{\sum_{v_k \in Out(v_j)} W_{jk}} PR(v_j)$$
where $d = 0.85$ is the damping factor.

### 4. Why is Negation Preservation Critical?
Standard NLP pipelines drop stopwords like `not`, `no`, `without`, `never`. In a contract sentence such as:
> *"The Consultant shall not disclose proprietary information without prior written approval."*
Stripping stopwords yields:
> *"Consultant disclose proprietary information prior written approval."*
This inverts the entire meaning! Our `LegalTextPreprocessor` explicitly preserves negation qualifiers throughout tokenization.

---

## 📄 License
This project is developed as an educational AI/NLP mini project.
