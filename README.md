#🧭 ATS-Savvy Career Coach — Naive RAG Chatbot

A Retrieval-Augmented Generation app that helps job-seekers craft ATS-friendly resumes, prepare interview answers, and target JD keywords—grounded entirely in a curated, practical corpus. Supports user uploads of Job Descriptions (JDs) or CVs to tailor guidance.

Why it stands out

Typical RAG demos index Wikipedia. This project targets career outcomes: ATS parsing, STAR bullet writing, interview mastery, and keyword targeting. It also ingests user JDs/CVs and uses a CrossEncoder reranker for precise, actionable answers.

Features

Multi-format ingestion: Generated PDF + MD/TXT + user uploads (PDF/TXT)

Embeddings: all-MiniLM-L6-v2 (HuggingFace)

Vector store: Persistent Chroma (./chroma_jobcoach)

Retrieval: Dense + CrossEncoder rerank (ms-marco-MiniLM-L-6-v2)

Prompting: Grounded answers with citations; admits “don’t know”

UI: Streamlit chat, dataset switcher (CV/Cover Letters, Interviews, ATS & Keywords, All)

Modes: Q&A and Summarization

JD-aware: Paste or upload your JD/CV for tailored, grounded advice

Export: Chat history → PDF

Quickstart
git clone <your-repo-url>
cd <repo>
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...                        # Windows PowerShell: $Env:OPENAI_API_KEY="sk-..."
streamlit run app.py


Open http://localhost:8501
 and try:

“Tailor my CV bullets to this JD: (paste JD)”

“What are ATS-safe formatting rules?”

“Give STAR prompts for handling a missed deadline.”

“Extract ATS keywords for a Sales Development Rep role.”

Tech Stack

LangChain (prompting, loaders)

HuggingFace Embeddings (all-MiniLM-L6-v2)

Chroma (persistent vector DB)

Sentence-Transformers CrossEncoder (MS MARCO MiniLM)

Streamlit (UI)

OpenAI Chat model (generation; configurable)

How it works

Ingestion & chunking

Generates a compact ATS PDF and loads MD/TXT guidance.

Accepts user uploads (JD/CV).

Splits with chunk_size=650, overlap=100 to preserve lists/STAR flow.

Embeddings & indexing

all-MiniLM-L6-v2 sentence embeddings.

Chroma persistence in ./chroma_jobcoach.

Retrieval & reranking

Dense top-k similarity → CrossEncoder reranks to top-m.

Boosts precision on answer-bearing passages (e.g., exact ATS rule or STAR recipe).

Prompting & generation

Strict grounding: cite sources; say “don’t know” when missing.

Q&A and Summarization templates.

Why reranking improves accuracy

Dense search maximizes recall but often returns near-matches. The CrossEncoder jointly attends to the (query, chunk), promoting the passage that directly answers the user’s need (e.g., “ATS: avoid tables” over generic resume tips). This reduces off-topic answers and clarifies what to write.

Example Q&A

Q: What are ATS-safe ways to format a resume?
A: Use a single column, avoid tables/text boxes, clear section headers, and mirror JD keywords in Skills and bullets. (ATS & Keywords — ATS_Savvy sections)

Q: Tailor bullets for SDR role (JD uploaded).
A: Highlight pipeline growth, CRM proficiency, MQL→SQL conversion, quota attainment, and call/email cadence improvements—each with metrics. (CV & Cover Letters + ATS & Keywords)

Limitations & improvements

Small demo corpus; bring your own PDFs/handbooks for better coverage.

OpenAI key required for generation; retrieval runs locally.

Add BM25 hybrid for sparse + dense gains.

Add inline source highlighting within generated text.

Contribute your own docs

Add PDFs to data/ and uploads/.

Update load_and_chunk_documents() to include new sources.

Rebuild index via the sidebar button.
