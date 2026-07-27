# Agri-Genius 🌾

**AI-Powered Agricultural Chatbot** — A RAG (Retrieval-Augmented Generation) chatbot that provides expert farming advice using a local vector database and LLM.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)

## Features

- **RAG-powered responses** — Retrieves relevant context from agricultural knowledge base before generating answers
- **PDF & Text ingestion** — Drop `.pdf` or `.txt` files into `data/` and ingest automatically
- **Local vector embeddings** — Uses `all-MiniLM-L6-v2` for fast, local embeddings via ChromaDB
- **Modern chat UI** — Dark-themed, glassmorphism design with animations and suggestion chips
- **OpenRouter integration** — Access 100+ LLMs through a single API

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, Flask |
| Vector Store | ChromaDB (local) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | OpenRouter API (any model) |
| Frontend | Vanilla HTML/CSS/JS |

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Ravjot-anand/Agri-Genius-Chatbot.git
cd Agri-Genius-Chatbot
pip install -r requirements.txt
```

### 2. Set API Key

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Get a free API key at [openrouter.ai/keys](https://openrouter.ai/keys).

### 3. Ingest Knowledge Base

```bash
# Add your PDFs/text files to data/ folder, then:
python ingest.py
```

### 4. Run

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Adding Your Own Data

Drop any `.pdf` or `.txt` agricultural documents into the `data/` folder and re-run:

```bash
python ingest.py
```

The pipeline will automatically discover, extract text, chunk, embed, and store all documents.

## Project Structure

```
AgriChatBot/
├── app.py                  # Flask backend with RAG endpoint
├── ingest.py               # Multi-format ingestion pipeline
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── .env                    # API key (not committed)
├── .gitignore              # Git ignore rules
├── data/                   # Knowledge base documents
│   └── agriculture_knowledge.txt
├── chroma_db/              # Vector embeddings (auto-generated)
└── templates/
    └── index.html          # Chat UI
```

## Deployment (Render)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service → Connect your GitHub repo
3. Render will auto-detect `render.yaml` and configure everything
4. Add `OPENROUTER_API_KEY` in Render's Environment settings

## License

MIT
