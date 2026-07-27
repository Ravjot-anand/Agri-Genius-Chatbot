# Agri-Genius: AI-Powered Agricultural Chatbot

Agri-Genius is a Retrieval-Augmented Generation (RAG) agricultural chatbot engineered to deliver precise, context-aware farming and agronomy advice. Built as a collaborative group project, the application combines custom document ingestion, vector similarity search using Chroma DB, and Large Language Model (LLM) orchestration via the OpenRouter API.

- **GitHub Repository:** [https://github.com/Ravjot-anand/Agri-Genius-Chatbot](https://github.com/Ravjot-anand/Agri-Genius-Chatbot)

---

## Role & Key Contributions

As part of this group initiative, my primary focus was designing the core system architecture and building the RAG pipeline.

### Key Contributions:
- **System Architecture Design:** Designed the end-to-end data flow spanning document parsing, chunking, vector storage, context retrieval, prompt construction, and server responses.
- **RAG & Ingestion Pipeline:** Implemented `ingest.py` using PyMuPDF and `RecursiveCharacterTextSplitter` to parse multi-format agricultural research (PDF and TXT) and compute vector embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Vector Database Integration:** Integrated Chroma DB for local persistent vector storage and fast cosine-similarity context retrieval.
- **Backend & LLM Integration:** Developed the Flask web server (`app.py`) interfacing with OpenRouter for model execution and built standard error handling routines.

---

## System Architecture

The following diagram illustrates the data ingestion and query execution flow of the Agri-Genius platform:

```
+-----------------------------------------------------------------------------------+
|                                INGESTION PIPELINE                                 |
|                                                                                   |
|  +--------------------+    +--------------------+    +-------------------------+  |
|  | Agricultural Data  | -> | PyMuPDF / Text     | -> | Recursive Character     |  |
|  | (PDFs & TXT)       |    | Loader             |    | Text Splitter           |  |
|  +--------------------+    +--------------------+    +-------------------------+  |
|                                                                   |               |
|                                                                   v               |
|  +--------------------+    +--------------------+    +-------------------------+  |
|  | Persistent         | <- | Local Embedding    | <- | Document Chunks         |  |
|  | Chroma DB Store    |    | (all-MiniLM-L6-v2) |    | (Overlap: 50, Size: 500)|  |
|  +--------------------+    +--------------------+    +-------------------------+  |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
|                                 RAG QUERY PIPELINE                                |
|                                                                                   |
|  +--------------------+    +--------------------+    +-------------------------+  |
|  | User Interface     | -> | Flask Server       | -> | Embed Query             |  |
|  | (HTML/CSS/JS)      |    | (/chat Endpoint)   |    | (all-MiniLM-L6-v2)      |  |
|  +--------------------+    +--------------------+    +-------------------------+  |
|                                                                   |               |
|                                                                   v               |
|  +--------------------+    +--------------------+    +-------------------------+  |
|  | LLM Response       | <- | OpenRouter API     | <- | Vector Similarity       |  |
|  | Rendered to User   |    | (Prompt + Context) |    | Search (Top-K Chunks)   |  |
|  +--------------------+    +--------------------+    +-------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## What I Learned

Building this project provided hands-on experience in practical AI engineering and RAG architecture design:

1. **Vector Databases & Embeddings:** Gained deep insight into how text embeddings are generated locally (`sentence-transformers`), stored, indexed, and queried using vector databases like Chroma DB.
2. **Retrieval-Augmented Generation (RAG):** Learned how to mitigate LLM hallucinations by dynamically injecting domain-specific context into prompts before inference.
3. **Document Processing at Scale:** Understood the nuances of document chunking, overlap strategies, and extracting clean textual content from multi-page PDF research papers.
4. **Backend Engineering & Integration:** Mastered local server configuration, environment variable management, and REST API context integration.

---

## Technology Stack

- **Language & Core Framework:** Python 3.10+, Flask
- **Vector Database:** Chroma DB (Local Persistent Mode)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Text Splitter & Loaders:** `langchain-text-splitters`, PyMuPDF (`fitz`)
- **LLM Interface:** OpenRouter API (OpenAI Python SDK)
- **Frontend:** Vanilla HTML5, CSS3, JavaScript

---

## Local Development Setup

### 1. Repository Setup
```bash
git clone https://github.com/Ravjot-anand/Agri-Genius-Chatbot.git
cd Agri-Genius-Chatbot
```

### 2. Environment Dependencies
```bash
pip install -r requirements.txt
```

### 3. API Configuration
Create a `.env` file in the project root:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Data Ingestion
Place your agricultural PDFs or text files inside the `data/` folder and execute the ingestion script:
```bash
python ingest.py
```

### 5. Launch Application
```bash
python app.py
```
Access the application locally at `http://127.0.0.1:5000`.

---

## Project Structure

```
Agri-Genius-Chatbot/
├── app.py                  # Flask web server and RAG chat handler
├── ingest.py               # Document discovery, PDF parsing, and Chroma DB ingestion
├── requirements.txt        # Python package dependencies
├── .env                    # Local environment secrets (excluded from git)
├── .gitignore              # Version control exclusions
├── data/                   # Knowledge repository (PDF and TXT research files)
│   └── agriculture_knowledge.txt
├── chroma_db/              # Generated vector store (excluded from git)
└── templates/
    └── index.html          # Web UI interface
```

---

## License

This project is licensed under the MIT License.
