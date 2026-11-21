# College Rulebook Q&A (Problem Statement 32)

A Retrieval-Augmented Generation (RAG) application that lets students ask questions about a college handbook PDF. It retrieves relevant text chunks with FAISS and asks Gemini to answer using only those chunks.

## Features
- Upload a college handbook PDF.
- Extract text page by page and split it into overlapping chunks.
- Create sentence embeddings with `all-MiniLM-L6-v2` and index them in FAISS.
- Retrieve the top 3 relevant chunks for each question.
- Generate a grounded answer with Gemini.
- Show retrieved source text and PDF page numbers.
- Return **"Not found in the uploaded handbook."** when the retrieved context is weak or the answer is absent.

## Requirements
- Python 3.10 or newer
- Internet connection for the first download of the embedding model and for Gemini API calls
- A Gemini API key from Google AI Studio
- A text-based/selectable-text PDF handbook (scanned PDFs need OCR first)

## Run locally
```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
streamlit run app.py
```

Open the local URL shown by Streamlit, enter your Gemini API key in the sidebar, upload the college handbook PDF, click **Process handbook**, and ask a question.

## Project workflow
1. Upload and read the handbook PDF.
2. Extract text from each page and retain its page number.
3. Split each page into chunks (900 characters with 150-character overlap).
4. Convert chunks into normalized sentence embeddings.
5. Store embeddings in a FAISS inner-product index.
6. Embed the student's question and retrieve the top 3 chunks.
7. Pass only the retrieved excerpts and a strict grounding prompt to Gemini.
8. Display the answer and source pages.

## Test plan
Use at least 10 questions based on the actual handbook. Suggested questions:
1. What is the minimum attendance requirement?
2. What happens if a student has attendance below the required level?
3. What are the rules for appearing in semester examinations?
4. What is the procedure for applying for a re-examination?
5. What are the rules for internal assessment?
6. What is the policy for late submission of assignments?
7. What are the rules about student identity cards?
8. What conduct is expected inside classrooms?
9. What is the procedure for requesting leave?
10. Whom should a student contact for an academic grievance?

These are test questions, not claims about your college's actual rules. Record the real output and check it against the cited handbook pages. Include screenshots of 3–5 actual runs in the submission.

## Important limitations
- Scanned/image-only PDFs may not produce extractable text; OCR is not included.
- Retrieval quality depends on handbook text quality and the embedding model.
- LLM output can still be wrong. Verify important rules against the cited page and the official college office.
- The API key is entered at runtime and is not stored in the source code.

## Security
Do not commit API keys, `.env` files, virtual environments, or private college documents to a public GitHub repository.
