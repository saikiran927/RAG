import asyncio
import json
import sqlite3
import numpy as np
import pandas as pd
import faiss
import os
from google import genai
from bs4 import BeautifulSoup
from loguru import logger
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")


SYSTEM_PROMPT = """You are a Analytics Vidya Python  Q&A assistant. Answer ONLY using the context is in this format provided below.
Rules:
- If the answer is not present in the context, respond with: "I don't have enough information to answer this question."
- Do NOT use any external knowledge outside the context.
- Keep your answer short, clear and on point.
- Do not repeat the question back.
- Do not make up or assume any information.
"""
client = genai.Client(api_key=GEMINI_API_KEY)


class AskRequest(BaseModel):
    query: str
    top_k_search_results: int = 5
    temperature: float = 0.3
    max_output_tokens: int = 1024

QUESTIONS_PATH = "Documents/Questions.csv"
ANSWERS_PATH = "Documents/Answers.csv"
DB_PATH = "qa.db"
FAISS_DIR = "./faiss"
FAISS_INDEX_PATH = f"{FAISS_DIR}/faiss_index.index"
FAISS_IDS_PATH = f"{FAISS_DIR}/faiss_ids.npy"
METADATA_PATH = f"{FAISS_DIR}/metadata.json"
os.makedirs(FAISS_DIR, exist_ok=True)


model = SentenceTransformer('./model/all-MiniLM-L6-v2')
EMBEDDING_DIM = model.get_embedding_dimension()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qa (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            question  TEXT NOT NULL,
            answer    TEXT NOT NULL,
            score     INTEGER,
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception as e:
        print(f"Startup error: {e}")
        raise
    yield


app = FastAPI(title="Analytics Vidya RAG", lifespan=lifespan)


def clean_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()


def embed(text: str) -> np.ndarray:
    return model.encode([text])[0].astype(np.float32)


def extract_qa() -> list[dict]:
    try:
        logger.info("Extracting Q&A from CSV files")

        df_questions = pd.read_csv(
            QUESTIONS_PATH,
            encoding="latin-1",
            usecols=["Id", "Score", "Title"],
        )
        df_answers = pd.read_csv(
            ANSWERS_PATH,
            encoding="latin-1",
            usecols=["ParentId", "Score", "Body"],
        )

        df_questions = df_questions[df_questions["Score"] > 0]

        df_answers = (
            df_answers[df_answers["Score"] > 0]
            .sort_values("Score", ascending=False)
            .drop_duplicates(subset=["ParentId"])
        )

        qa = (
            df_questions.merge(df_answers, left_on="Id", right_on="ParentId")
            .rename(columns={"Title": "Question", "Body": "Answer", "Score_x": "Score"})
            [["Question", "Answer", "Score"]]
        )

        qa = qa.dropna(subset=["Question", "Answer"])
        qa = qa.sort_values("Score", ascending=False)
        logger.info(f"Fetched {len(qa)} Q&A pairs {EMBEDDING_DIM}")
    except Exception as e:
        logger.exception(f"Error extracting Q&A: {e}")
        raise
    return qa.to_dict(orient="records")


@app.post("/ingest", summary="Process CSVs, create embeddings, store in SQLite and FAISS")
def ingest():
    try:
        records = extract_qa()

        conn = sqlite3.connect(DB_PATH)
        logger.info("Connected to SQLite database")
        conn.execute("DELETE FROM qa")

        embeddings = []
        sqlite_ids = []

        for ind, record in enumerate(records):
            question = record.get("Question", "").strip()
            answer = clean_html(record.get("Answer", "")).strip()
            score = int(record.get("Score") or 0)

            if not question or not answer:
                logger.warning(f"Skipping record {ind + 1} — missing question or answer")
                continue

            vector = embed(question)
            embedding_blob = vector.tobytes()

            cursor = conn.execute(
                "INSERT INTO qa (question, answer, score, embedding) VALUES (?, ?, ?, ?)",
                (question, answer, score, embedding_blob),
            )
            sqlite_ids.append(cursor.lastrowid)
            embeddings.append(vector)
            logger.info(f"Processed record {ind + 1}/{len(records)}")
 
        conn.commit()
        conn.close()

        # Build FAISS index
        logger.info(f"Building FAISS index... {EMBEDDING_DIM}")
        vectors = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(vectors)
        print(f"FAISS index built with {index.ntotal} vectors")
        if index.ntotal != len(sqlite_ids):
            logger.error("Mismatch between FAISS index size and SQLite IDs")
            raise ValueError("FAISS index size does not match number of SQLite IDs")

        faiss.write_index(index, FAISS_INDEX_PATH)
        np.save(FAISS_IDS_PATH, np.array(sqlite_ids, dtype=np.int64))

        metadata = {
            str(faiss_pos): sqlite_id
            for faiss_pos, sqlite_id in enumerate(sqlite_ids)
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"FAISS index saved with {index.ntotal} vectors → {FAISS_INDEX_PATH}")
        logger.info(f"Metadata saved with {len(metadata)} entries → {METADATA_PATH}")

        return JSONResponse(content={
            "message": "Ingestion complete",
            "records_stored": len(sqlite_ids),
            "faiss_vectors": index.ntotal,
        })

    except FileNotFoundError as e:
        logger.exception(f"CSV file not found: {e}")
        raise HTTPException(status_code=404, detail=f"CSV file not found: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def search(query: str, top_k: int = 10) -> list[dict]:
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("FAISS index not found. Run /ingest first.")

    # Load FAISS index and metadata
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # Embed and normalize query vector for IndexFlatIP (cosine similarity)
    query_vector = embed(query).reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # Search top_k nearest vectors
    distances, positions = index.search(query_vector, top_k)
    logger.info(f"FAISS search returned positions: {positions[0]}, distances: {distances[0]}")

    # Map FAISS positions → SQLite ids
    sqlite_ids = [
        metadata[str(pos)]
        for pos in positions[0]
        if pos != -1 and str(pos) in metadata
    ]

    if not sqlite_ids:
        return []

    # Fetch matching documents from SQLite
    placeholders = ",".join("?" * len(sqlite_ids))
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        f"SELECT id, question, answer, score FROM qa WHERE id IN ({placeholders})",
        sqlite_ids,
    ).fetchall()
    conn.close()

    return [
        {"id": r[0], "question": r[1], "answer": r[2], "score": r[3]}
        for r in rows
    ]


@app.post("/ask", summary="Ask a question and get an LLM answer grounded in context")
async def ask(request: AskRequest):
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Retrieve top 10 relevant documents from FAISS + SQLite
        results = await asyncio.to_thread(search, query, request.top_k_search_results)
        if not results:
            return JSONResponse(content={
                "query": query,
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
            })

        # Build context from retrieved documents
        answers = [r["answer"] for r in results]
        
        context = "\n".join(answers)

        prompt = f"""
                User Question: {query}
                Context:
                {context}

                """

        # logger.info(f"Sending prompt to Gemini for query: {query}")
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
            )
        )
        answer = response.text.strip()
        logger.info(f"Gemini response received {response.text.strip()}")

        return JSONResponse(content={
            "query": query,
            "answer": answer,
            # "sources": [{"id": r["id"], "question": r["question"]} for r in results],
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=7000, reload=False)
