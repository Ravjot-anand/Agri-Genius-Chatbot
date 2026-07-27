"""
app.py — Agri-Genius Flask Backend

Serves the chat UI and exposes a /chat endpoint that performs RAG:
  1. Embeds the user query
  2. Retrieves relevant context from ChromaDB
  3. Calls OpenRouter LLM with context-augmented prompt
  4. Returns the response as JSON
"""

import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import chromadb
from openai import OpenAI

# ── Memory Optimization for Low-RAM Environments (Render 512MB) ─────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer

# ── Load .env file ─────────────────────────────────────────────────────────
load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "agri_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
TOP_K = 3  # Number of context chunks to retrieve

# ── Initialize Flask ───────────────────────────────────────────────────────
app = Flask(__name__)

# ── Initialize Services (loaded once at startup) ──────────────────────────
print("[*] Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("[*] Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
print(f"[OK] ChromaDB connected -- {collection.count()} documents in '{COLLECTION_NAME}'")

# OpenRouter client (initialized per-request to allow env var changes)
def get_llm_client() -> OpenAI:
    """Create an OpenAI client configured for OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Set it with: $env:OPENROUTER_API_KEY = 'sk-or-v1-...'"
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# ── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Agri-Genius, an expert agricultural advisor chatbot. You provide accurate, practical, and helpful advice about farming, crops, soil management, pest control, irrigation, organic farming, and modern agricultural technologies.

INSTRUCTIONS:
- Answer questions based PRIMARILY on the provided context. If the context contains relevant information, use it.
- If the context doesn't fully cover the question, you may supplement with your general agricultural knowledge, but clearly indicate when you're doing so.
- Keep answers concise but thorough. Use bullet points and structured formatting when helpful.
- Include specific numbers, varieties, and recommendations when available in the context.
- If you genuinely don't know something, say so honestly rather than guessing.
- Always be encouraging and supportive to farmers seeking advice.

CONTEXT FROM KNOWLEDGE BASE:
{context}
"""


# ── Routes ─────────────────────────────────────────────────────────────────
def query_rag(user_message: str) -> str:
    """Core RAG logic: embed query, fetch ChromaDB context, call OpenRouter LLM."""
    query_embedding = embedder.encode(user_message).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )
    context_chunks = results["documents"][0] if results["documents"] else []
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."

    system_message = SYSTEM_PROMPT.format(context=context)
    client = get_llm_client()
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return completion.choices[0].message.content


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the chat interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """RAG-powered chat endpoint."""
    try:
        data = request.get_json()
        if not data or not data.get("message", "").strip():
            return jsonify({"error": "Message cannot be empty."}), 400

        user_message = data["message"].strip()
        assistant_reply = query_rag(user_message)

        return jsonify({
            "response": assistant_reply,
            "sources": TOP_K,
        })

    except ValueError as e:
        print(f"[ERROR] /chat (config): {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
            error_msg = "API key issue: " + error_msg
        elif "model" in error_msg.lower():
            error_msg = "Model issue: " + error_msg
        return jsonify({"error": f"{error_msg}"}), 500


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[*] Agri-Genius server starting...")
    print("   Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host="127.0.0.1", port=5000)


