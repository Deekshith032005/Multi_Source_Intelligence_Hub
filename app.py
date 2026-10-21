
import os
import time
import uuid

import chromadb
import pymupdf
from dotenv import load_dotenv
from flask import Flask, render_template, request
from google import genai
from sentence_transformers import SentenceTransformer

from db import get_db_connection


# --------------------------------------------------
# 1. Application setup
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# Load the embedding model once at startup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Save document embeddings on disk
chroma = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, "chroma_data")
)

collection = chroma.get_or_create_collection(
    name="pdf_passages"
)

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
gemini = genai.Client(api_key=api_key) if api_key else None

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# --------------------------------------------------
# 2. Text processing
# --------------------------------------------------

def split_text(text, chunk_size=700, overlap=100):
    """Split text into smaller overlapping passages."""
    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def get_active_document():
    """Return the filename of the currently stored PDF."""
    result = collection.get(
        limit=1,
        include=["metadatas"],
    )

    metadatas = result.get("metadatas") or []

    if metadatas and metadatas[0]:
        return metadatas[0].get("source")

    return None


# --------------------------------------------------
# 3. Upload and index a PDF
# --------------------------------------------------

def load_pdf(pdf_file):
    """Extract text, create embeddings and save passages."""
    document = pymupdf.open(
        stream=pdf_file.read(),
        filetype="pdf",
    )

    chunks = []
    metadata = []

    try:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text().strip()

            if not page_text:
                continue

            for chunk in split_text(page_text):
                chunks.append(chunk)
                metadata.append({
                    "page": page_number,
                    "source": pdf_file.filename,
                })
    finally:
        document.close()

    if not chunks:
        raise ValueError(
            "No readable text was found. "
            "The PDF may be scanned or empty."
        )

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False,
    ).tolist()

    # Replace previously stored passages
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadata,
        embeddings=embeddings,
    )

    return len(chunks)


# --------------------------------------------------
# 4. Ask Gemini with temporary-error handling
# --------------------------------------------------

def generate_answer(prompt):
    """Ask Gemini and retry once after a temporary 503 error."""
    if gemini is None:
        raise ValueError(
            "Gemini API key is missing. Check your .env file."
        )

    for attempt in range(2):
        try:
            response = gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty answer."
                )

            return response.text

        except Exception as exc:
            message = str(exc).upper()

            is_temporary = (
                "503" in message
                or "UNAVAILABLE" in message
            )

            if is_temporary and attempt == 0:
                time.sleep(2)
                continue

            raise


# --------------------------------------------------
# 5. Save chat history to MySQL
# --------------------------------------------------

def save_chat_history(question, answer, source_document):
    """Save a successful question and answer in MySQL."""
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO chat_history
                (question, answer, source_document)
            VALUES (%s, %s, %s)
            """,
            (question, answer, source_document),
        )

        connection.commit()

    finally:
        if connection.is_connected():
            if "cursor" in locals():
                cursor.close()
            connection.close()


# --------------------------------------------------
# 6. Main webpage
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    error = None
    sources = []
    document_name = get_active_document()

    if request.method == "POST":
        pdf_file = request.files.get("pdf")
        question = request.form.get("question", "").strip()

        try:
            # Process a new PDF only when one is selected
            if pdf_file and pdf_file.filename:
                if not pdf_file.filename.lower().endswith(".pdf"):
                    raise ValueError("Please upload a PDF file.")

                load_pdf(pdf_file)
                document_name = get_active_document()

            # Reuse the previously uploaded PDF
            if collection.count() == 0:
                raise ValueError(
                    "No PDF is loaded. Upload a PDF first."
                )

            if not question:
                raise ValueError("Please enter a question.")

            # Convert the question into an embedding
            question_embedding = embedding_model.encode(
                [question],
                show_progress_bar=False,
            ).tolist()

            # Retrieve relevant PDF passages
            results = collection.query(
                query_embeddings=question_embedding,
                n_results=min(5, collection.count()),
            )

            retrieved_chunks = results["documents"][0]
            retrieved_metadata = results["metadatas"][0]

            context_parts = []

            for chunk, meta in zip(
                retrieved_chunks,
                retrieved_metadata,
            ):
                sources.append(meta["page"])
                context_parts.append(
                    f"Page {meta['page']}:\n{chunk}"
                )

            context = "\n\n".join(context_parts)

            prompt = f"""
You are answering questions about an uploaded PDF.

Rules:
- Use only the supplied PDF passages.
- Answer the question directly and clearly.
- If the passages do not contain the answer, say so.
- Do not invent facts.
- Include the relevant PDF page number in your answer.

PDF passages:
{context}

Question: {question}
"""

            # Generate the answer
            answer = generate_answer(prompt)

            sources = sorted(set(sources))
            document_name = get_active_document()

            # Save the successful answer to MySQL.
            # A database failure should not hide the answer.
            try:
                save_chat_history(
                    question,
                    answer,
                    document_name,
                )
            except Exception:
                app.logger.exception(
                    "Could not save chat history to MySQL"
                )
                error = (
                    "Your answer was generated, but chat history "
                    "could not be saved to MySQL."
                )

        except ValueError as exc:
            error = str(exc)

        except Exception as exc:
            app.logger.exception(
                "PDF question-answering failed"
            )

            message = str(exc).upper()

            if "503" in message or "UNAVAILABLE" in message:
                error = (
                    "Gemini is temporarily busy. Please wait a little "
                    "and submit your question again. Your PDF is saved; "
                    "you do not need to upload it again."
                )
            elif "429" in message or "RESOURCE_EXHAUSTED" in message:
                error = (
                    "Gemini's usage limit has been reached. "
                    "Please wait or check your API quota."
                )
            else:
                error = (
                    "Something went wrong. Check the VS Code "
                    "terminal for the exact error."
                )

    return render_template(
        "index.html",
        answer=answer,
        error=error,
        sources=sources,
        document_name=document_name,
    )


# --------------------------------------------------
# 7. Start the local server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False,
    )
