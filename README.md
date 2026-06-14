# Analytics Vidya RAG — Python Q&A API

A Retrieval-Augmented Generation (RAG) pipeline built on top of a Stack Overflow Python Q&A dataset. It ingests questions and answers from CSV files, creates semantic embeddings, stores them in a FAISS vector index and SQLite database, and serves an API that answers user queries using Google Gemini grounded strictly in the retrieved context.

---

## Architecture

```
CSV Files (Questions + Answers)
        ↓
   Preprocessing (merge, clean HTML, remove NaN)
        ↓
   Embedding (sentence-transformers all-MiniLM-L6-v2)
        ↓
   ┌──────────────┐     ┌─────────────────┐
   │  SQLite DB   │     │  FAISS Index    │
   │  (qa.db)     │     │  (IndexFlatIP)  │
   └──────────────┘     └─────────────────┘
        ↓                       ↓
        └──────── /ask ─────────┘
                   ↓
            Google Gemini LLM
                   ↓
             Grounded Answer
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector Database | FAISS (`IndexFlatIP` — cosine similarity) |
| Document Store | SQLite3 |
| LLM | Google Gemini (`gemini-2.0-flash`) |
| HTML Cleaning | BeautifulSoup4 |
| Logging | Loguru |

---

## Project Structure

```
Analytics_Vidya/
├── main.py                  # FastAPI app — all endpoints
├── pyproject.toml           # Dependencies
├── Documents/
│   ├── Questions.csv        # Stack Overflow questions
│   └── Answers.csv          # Stack Overflow answers
├── model/
│   └── all-MiniLM-L6-v2/   # Saved embedding model weights
├── faiss/
│   ├── faiss_index.index    # FAISS vector index (created by /ingest)
│   ├── faiss_ids.npy        # FAISS position → SQLite id mapping
│   └── metadata.json        # FAISS position → SQLite id (JSON)
└── qa.db                    # SQLite database
```

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Save the embedding model locally (run once)

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save("./model/all-MiniLM-L6-v2")
```

### 3. Set the Gemini API key

```powershell
# PowerShell
$env:GEMINI_API_KEY = "your-api-key-here"
```

```bash
# Linux / Mac
export GEMINI_API_KEY="your-api-key-here"
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. Run the server

```bash
python main.py
```

Server starts at `http://localhost:7000`. Swagger UI available at `http://localhost:7000/docs`.

---

## API Endpoints

### `POST /ingest`

Reads the CSV files, preprocesses the data, creates embeddings for all questions, stores everything in SQLite and builds the FAISS index. **Run this once before using `/search` or `/ask`.**

**What it does internally:**
1. Loads `Questions.csv` and `Answers.csv`
2. Filters out questions and answers with score ≤ 0
3. Keeps only the highest-scored answer per question
4. Merges questions with their best answers
5. Strips HTML tags from answer bodies using BeautifulSoup
6. Drops rows with missing question or answer
7. Creates a 384-dimensional embedding for each question using `all-MiniLM-L6-v2`
8. Inserts all records into SQLite (`qa.db`)
9. Builds a `faiss.IndexFlatIP` index from all embeddings
10. Saves the FAISS index to `faiss/faiss_index.index`
11. Saves the FAISS position → SQLite id mapping to `faiss/metadata.json`

**Request:** No body required.

**Response:**
```json
{
  "message": "Ingestion complete",
  "records_stored": 38142,
  "faiss_vectors": 38142
}
```

**Notes:**
- Re-running `/ingest` clears the existing SQLite table and rebuilds the FAISS index from scratch.
- This is a slow operation — it embeds every question one by one. Expect several minutes for large datasets.

---

### `GET /search?q=<query>`

Performs a semantic search over the FAISS index and returns the top 10 most relevant Q&A pairs without calling the LLM. Useful for debugging retrieval quality before running `/ask`.

**What it does internally:**
1. Embeds the query using the same `all-MiniLM-L6-v2` model
2. L2-normalizes the query vector (required for cosine similarity with `IndexFlatIP`)
3. Searches the FAISS index for the top 10 nearest vectors
4. Maps FAISS positions back to SQLite ids using `metadata.json`
5. Fetches the full Q&A records from SQLite

**Query parameter:**

| Parameter | Type | Description |
|---|---|---|
| `q` | string | The search query text |

**Example:**
```
GET http://localhost:7000/search?q=How to reverse a list in Python
```

**Response:**
```json
{
  "query": "How to reverse a list in Python",
  "count": 10,
  "results": [
    {
      "id": 1,
      "question": "How do you reverse a list in Python?",
      "answer": "Use list[::-1] or list.reverse()...",
      "score": 2341
    }
  ]
}
```

---

### `POST /ask`

The main RAG endpoint. Takes a user question, retrieves the top 10 relevant Q&A pairs from FAISS, and sends them as context to Google Gemini to generate a grounded, concise answer.

**What it does internally:**
1. Validates the query is not empty
2. Calls `search()` internally to retrieve top 10 relevant documents
3. Concatenates all retrieved answers into a context block
4. Builds a prompt combining the context and user question
5. Sends the prompt to `gemini-2.0-flash` with a strict system instruction
6. Returns the generated answer

**System prompt rules enforced on the LLM:**
- Answer ONLY using the provided context
- If the answer is not in the context, respond: *"I don't have enough information to answer this question."*
- No external knowledge — no assumptions or made-up information
- Keep answers short and on point

**Request body:**
```json
{
  "query": "What does the yield keyword do in Python?"
}
```

**Response:**
```json
{
  "query": "What does the yield keyword do in Python?",
  "answer": "The yield keyword is used to create a generator function. Instead of returning a value and terminating, yield pauses the function and saves its state, returning a value to the caller each time it is iterated."
}
```

**Error responses:**

| Status | Reason |
|---|---|
| `400` | Empty query |
| `404` | FAISS index not found — run `/ingest` first |
| `500` | Gemini API error or internal failure |

---

### `GET /extract`

Returns all stored Q&A records from the SQLite database. Useful for inspecting what was ingested.

**Request:** No parameters required.

**Response:**
```json
{
  "count": 38142,
  "data": [
    {
      "id": 1,
      "question": "How can I find the full path to a font from its display name on a Mac?",
      "answer": "Open up a terminal and type...",
      "score": 21
    }
  ]
}
```

**Note:** Returns all records — for large datasets this response can be very large. Use `/search` for targeted lookups.

---

## Usage Flow

```
1. POST /ingest          ← run once to populate DB and build FAISS index
2. GET  /search?q=...    ← optional: verify retrieval quality
3. POST /ask             ← ask questions, get grounded LLM answers
4. GET  /extract         ← inspect all stored records
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key from AI Studio |

---

## Dependencies

```
pandas
numpy
fastapi
uvicorn[standard]
sentence-transformers
faiss-cpu
google-generativeai
google-genai
beautifulsoup4
loguru
pydantic
requests
```
