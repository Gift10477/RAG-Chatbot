# app.py — Full workable Streamlit RAG app with robust OPENAI key handling
# NOTE: requires `pysqlite3-binary` installed in the same env as Streamlit.
# Install: pip install pysqlite3-binary

# ------------------ Monkeypatch to ensure modern sqlite is used ------------------
# Must run BEFORE any import that could trigger chromadb/sqlite3 to load.
try:
    # Try to use the pysqlite3 module (installed via pysqlite3-binary)
    import pysqlite3 as _pysqlite3  # this name is provided by pysqlite3-binary
    import sys

    # Only replace if the stdlib sqlite3 isn't already the desired implementation
    try:
        import sqlite3 as _stdlib_sqlite
        std_ver = getattr(_stdlib_sqlite, "sqlite_version", None)
    except Exception:
        std_ver = None

    # Replace stdlib sqlite3 module with pysqlite3 so downstream imports see a newer sqlite
    if std_ver is None or (isinstance(std_ver, str) and tuple(map(int, std_ver.split("."))) < (3, 35, 0)):
        sys.modules["sqlite3"] = _pysqlite3
except Exception:
    # If anything fails, continue — downstream imports will raise the original, informative error.
    pass

# ------------------ Regular imports (unchanged) ------------------
import os
import io
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st

# LangChain core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# LLM
from langchain_openai import ChatOpenAI  # requires api_key param

# Reranker
from sentence_transformers import CrossEncoder

# PDF generation for corpus + export
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# Optional dotenv
try:
    from dotenv import load_dotenv  # pip install python-dotenv
except Exception:
    load_dotenv = None

# ---------- Content corpus ----------
CV_PLAYBOOK = """
# ATS-Savvy CV & Cover Letter Playbook

## CV Structure (ATS-Friendly)
- Header: Name | Email | Phone | City | LinkedIn/GitHub (plain text, not images/icons)
- Professional Summary: 2–3 lines with role target + core strengths + 1 quantified outcome
- Skills (Keywords): Group by category (e.g., Sales | CRM | Analytics)
- Experience: Reverse-chronological; 3–6 bullets per role; each bullet follows STAR and includes a metric
- Education/Certs: Degree, institution, year; relevant certs
- Formatting: Single column, consistent section headers, no tables/text boxes

## STAR Bullet Template
- Situation/Task: brief context
- Action: what you did (strong verb)
- Result: quantified outcome (+%/-%, time saved, revenue, cost)
Example:
- Grew SME pipeline by 38% in 2 quarters by launching a lead-scoring rubric and weekly SDR standups (HubSpot, SQL).

## Action Verbs (Mix)
Led, Built, Launched, Automated, Reduced, Optimized, Negotiated, Coordinated, Analyzed, Designed, Implemented, Improved

## Cover Letter Outline (ATS-Safe)
- Hook: 1–2 lines tying your impact to role
- Why Me: 2–3 bullets mapping your achievements to JD must-haves
- Why This Company: 1–2 lines referencing mission/product
- Close: availability + call to action (interview)
"""

INTERVIEW_KIT = """
# Interview Mastery Kit

## Behavioral Questions (Use STAR)
- "Tell me about a time you handled a difficult stakeholder."
- "Describe a project you led end-to-end."
- "A time you failed—what changed afterward?"
Tips:
- Keep answers ~90 seconds.
- Lead with outcome and metrics; then context.

## Questions To Ask Interviewers
- How do you measure success for this role in 6 months?
- What are the top challenges the team faces?

## Salary Conversation
- Research a range; anchor with value.
- If asked early: "Happy to discuss; may I learn more about scope and impact first?"
"""

ATS_GUIDE = """
# ATS & Keywords Guide

## How ATS Parses
- Prefers simple layouts; avoid tables/columns/text boxes/images
- Uses section headers to find experience/skills
- Extracts exact keywords from JD

## Keyword Targeting
- Mirror critical JD terms in Skills and in relevant bullets
- For Sales roles: pipeline, quota, CRM, MQL/SQL, churn, close rate, CAC, LTV
- For Software roles: Python, React, APIs, CI/CD, Docker, Postgres, microservices
- For Ops roles: SOPs, KPIs, process mapping, lean, SLA, backlog, on-time delivery
"""

EA_MARKET_NOTES = """
# East Africa Job Search Notes (Concise, Neutral)

## Networking & Applications
- LinkedIn and company websites are primary; local boards vary by sector
- Referrals and alumni groups significantly improve response rate
- Keep contact details local and current

## CV Norms
- 1–2 pages; quantify outcomes
- Certifications can be impactful for early-career candidates
"""

GENERATED_PDF_NAME = "ATS_Resume_CheatSheet.pdf"

# ---------- Utilities ----------


def make_pdf(path: str, text: str, title="ATS Resume Cheat Sheet"):
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    margin = 2*cm
    textobj = c.beginText(margin, height - margin)
    textobj.setFont("Times-Roman", 12)
    textobj.textLine(title)
    textobj.textLine("")
    for line in text.strip().splitlines():
        if textobj.getY() < margin:
            c.drawText(textobj)
            c.showPage()
            textobj = c.beginText(margin, height - margin)
            textobj.setFont("Times-Roman", 12)
        textobj.textLine(line)
    c.drawText(textobj)
    c.showPage()
    c.save()

def write_text_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip())

def _load_env_file():
    """Load .env sitting next to this file if python-dotenv is installed."""
    if load_dotenv is None:
        return
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

def _normalize_key(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    key = key.strip().strip('"').strip("'")
    return key or None

def resolve_openai_api_key() -> Optional[str]:
    """
    Priority:
      1) Key pasted into Streamlit sidebar this session (st.session_state["OPENAI_API_KEY_INPUT"])
      2) st.secrets["OPENAI_API_KEY"]
      3) .env file (if present)
      4) OS environment variable OPENAI_API_KEY
    """
    # 1) Session key
    k = _normalize_key(st.session_state.get("OPENAI_API_KEY_INPUT"))
    if k:
        return k

    # 2) Streamlit secrets
    try:
        k = _normalize_key(st.secrets.get("OPENAI_API_KEY"))
        if k:
            return k
    except Exception:
        pass

    # 3) .env
    _load_env_file()

    # 4) OS env
    return _normalize_key(os.getenv("OPENAI_API_KEY"))


def load_and_chunk_documents(dataset_choice: str, uploaded_docs: List[Document]) -> List[Document]:
    """
    dataset_choice: 'CV & Cover Letters', 'Interviews', 'ATS & Keywords', 'All'
    uploaded_docs: preloaded docs from user uploads (PDF/TXT)
    """
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", GENERATED_PDF_NAME)
    make_pdf(pdf_path, CV_PLAYBOOK)

    cv_file = os.path.join("data", "cv_playbook.md")
    int_file = os.path.join("data", "interview_kit.md")
    ats_file = os.path.join("data", "ats_guide.md")
    ea_file = os.path.join("data", "ea_market_notes.md")
    write_text_file(cv_file, CV_PLAYBOOK)
    write_text_file(int_file, INTERVIEW_KIT)
    write_text_file(ats_file, ATS_GUIDE)
    write_text_file(ea_file, EA_MARKET_NOTES)

    pdf_loader = PyPDFLoader(pdf_path)
    cv_loader = TextLoader(cv_file, encoding="utf-8")
    int_loader = TextLoader(int_file, encoding="utf-8")
    ats_loader = TextLoader(ats_file, encoding="utf-8")
    ea_loader = TextLoader(ea_file, encoding="utf-8")

    chosen_loaders = []
    if dataset_choice in ("CV & Cover Letters", "All"):
        chosen_loaders.extend([pdf_loader, cv_loader])
    if dataset_choice in ("Interviews", "All"):
        chosen_loaders.append(int_loader)
    if dataset_choice in ("ATS & Keywords", "All"):
        chosen_loaders.extend([ats_loader, ea_loader])

    docs: List[Document] = []
    for loader in chosen_loaders:
        docs.extend(loader.load())

    docs.extend(uploaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    for d in chunks:
        src = (d.metadata or {}).get("source", "").lower()
        if "cv_playbook" in src or GENERATED_PDF_NAME.lower() in src:
            d.metadata["dataset"] = "CV & Cover Letters"
        elif "interview_kit" in src:
            d.metadata["dataset"] = "Interviews"
        elif "ats_guide" in src or "ea_market_notes" in src:
            d.metadata["dataset"] = "ATS & Keywords"
        elif "upload_" in src:
            d.metadata["dataset"] = "User Upload"
        else:
            d.metadata["dataset"] = "General"
    return chunks

def create_embeddings_and_store(
    chunks: List[Document],
    persist_dir: str = "./chroma_jobcoach",
    collection_name: str = "ats_job_coach",
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

@st.cache_resource(show_spinner=False)
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_and_rerank(vectordb: Chroma, query: str, k: int = 8, m: int = 3) -> List[Document]:
    initial = vectordb.similarity_search(query, k=k)
    pairs = [(query, d.page_content) for d in initial]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(initial, scores), key=lambda x: x[1], reverse=True)[:m]
    return [d for d, _ in ranked]

# ---------- Prompts & LLM ----------

SYSTEM_TMPL = """You are a careful career assistant that answers ONLY using the provided context.
If missing info, say you don't know. Keep answers ATS-safe and cite sources (dataset + file).
Support English or Kiswahili.
Context:
{context}
"""

QA_TMPL = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Using only the context below, answer the user's question.\n\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer (with citations):"
    ),
)

SUM_TMPL = PromptTemplate(
    input_variables=["context"],
    template=(
        "Summarize the key guidance from the context into concise, actionable bullet points "
        "for a job seeker. Keep it grounded and ATS-safe.\n\n{context}\n\nSummary:"
    ),
)

def llm_client(model_name: str = "gpt-4o-mini", temperature: float = 0.7):
    api_key = resolve_openai_api_key()
    if not api_key:
        st.warning("No OpenAI API key available. Add it in the sidebar or set OPENAI_API_KEY.")
        raise RuntimeError("Missing OPENAI_API_KEY for ChatOpenAI.")
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

def _extract_llm_text(resp) -> str:
    """
    LangChain LLM responses come in different shapes depending on version.
    Try to extract the human-facing text in a few common patterns.
    """
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "content"):
        return getattr(resp, "content") or ""
    if hasattr(resp, "generations"):
        try:
            gen = resp.generations[0][0]
            if hasattr(gen, "text"):
                return gen.text
            if hasattr(gen, "message"):
                msg = gen.message
                if hasattr(msg, "content"):
                    return msg.content
        except Exception:
            pass
    if isinstance(resp, list) and resp:
        first = resp[0]
        if hasattr(first, "content"):
            return first.content
        if isinstance(first, dict):
            return first.get("text") or first.get("content") or str(first)
    return str(resp)

def generate_answer(question: str, docs: List[Document], mode: str, model_name: str) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        blocks.append(f"[{i}] ({meta.get('dataset','Unknown')}) {meta.get('source','N/A')}\n{d.page_content}")
    ctx = "\n\n---\n\n".join(blocks)
    system_msg = SystemMessage(content=SYSTEM_TMPL.format(context=ctx))
    user_prompt = QA_TMPL.format(context=ctx, question=question) if mode == "Q&A" else SUM_TMPL.format(context=ctx)
    llm = llm_client(model_name=model_name)

    # Try a few call patterns depending on langchain version
    try:
        # preferred in some LangChain versions
        resp = llm([system_msg, HumanMessage(content=user_prompt)])
    except TypeError:
        # fallback to .invoke if present
        try:
            resp = llm.invoke([system_msg, HumanMessage(content=user_prompt)])
        except Exception as e:
            # last-resort: call low-level __call__
            resp = llm.__call__([system_msg, HumanMessage(content=user_prompt)])
    text = _extract_llm_text(resp)
    return text

def export_chat_to_pdf(history: List[Tuple[str, str]]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 2*cm
    t = c.beginText(margin, height - margin)
    t.setFont("Times-Roman", 11)
    for role, msg in history:
        block = f"{role.upper()}:\n{msg}\n\n"
        for line in block.splitlines():
            if t.getY() < margin:
                c.drawText(t)
                c.showPage()
                t = c.beginText(margin, height - margin)
                t.setFont("Times-Roman", 11)
            t.textLine(line)
    c.drawText(t); c.showPage(); c.save(); buf.seek(0)
    return buf.read()

def load_user_uploads(uploaded_files) -> List[Document]:
    docs = []
    if not uploaded_files:
        return docs
    up_dir = "uploads"
    os.makedirs(up_dir, exist_ok=True)
    for f in uploaded_files:
        path = os.path.join(up_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.read())
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            loaded = loader.load()
        else:
            loader = TextLoader(path, encoding="utf-8")
            loaded = loader.load()
        for d in loaded:
            d.metadata["dataset"] = "User Upload"
            d.metadata["source"] = f"upload_{os.path.basename(path)}"
        docs.extend(loaded)
    return docs


def run_app():
    st.set_page_config(page_title="ATS-Savvy Career Coach (RAG)", page_icon="🧭", layout="wide")
    st.title(" ATS-Savvy Career Coach — Naive RAG")
    

    with st.sidebar:
        st.header("Settings")
        st.subheader("API Key")
        st.text("Paste your OpenAI API key for this session (kept only in this session).")
        key_input = st.text_input("OPENAI_API_KEY", type="password", help="Paste your OpenAI key (sk-...)", value="")
        if key_input:
            st.session_state["OPENAI_API_KEY_INPUT"] = key_input
            os.environ["OPENAI_API_KEY"] = key_input
            st.success("API key set for this session.")
        dataset_choice = st.selectbox("Dataset", ["All", "CV & Cover Letters", "Interviews", "ATS & Keywords"])
        mode = st.radio("Mode", ["Q&A", "Summarization"])
        model_name = st.text_input("OpenAI model", value="gpt-4o-mini")
        top_k = st.slider("Retrieve k", 4, 12, 8)
        top_m = st.slider("Rerank to m", 1, 5, 3)
        st.markdown("---")

        uploaded_files = st.file_uploader(
            "Upload Job Description or your CV (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True
        )

        if st.button("Rebuild Index"):
            import shutil
            if os.path.exists("./chroma_jobcoach"):
                shutil.rmtree("./chroma_jobcoach")
            st.session_state.pop("vectordb", None)
            st.success("Index cleared. It will rebuild on next query.")

        if st.button("Export chat as PDF"):
            hist = st.session_state.get("history", [])
            pdf_bytes = export_chat_to_pdf(hist)
            st.download_button("Download PDF", data=pdf_bytes, file_name="chat_history.pdf", mime="application/pdf")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    uploaded_docs = load_user_uploads(uploaded_files)

    rebuild_needed = (
        ("vectordb" not in st.session_state)
        or (st.session_state.get("last_dataset") != dataset_choice)
        or (st.session_state.get("last_upload_count") != len(uploaded_docs))
    )

    if rebuild_needed:
        with st.spinner("Loading & chunking documents..."):
            chunks = load_and_chunk_documents(dataset_choice, uploaded_docs)
        with st.spinner("Embedding & indexing (Chroma)..."):
            vectordb = create_embeddings_and_store(chunks)
        st.session_state["vectordb"] = vectordb
        st.session_state["last_dataset"] = dataset_choice
        st.session_state["last_upload_count"] = len(uploaded_docs)

    st.write("**Try:**")
    st.code(
        "Tailor my CV bullets to this JD: (paste JD)\n"
        "— or —\n"
        "What are ATS-safe ways to format a resume?\n"
        "— or —\n"
        "Give STAR prompts for handling a missed deadline."
    )

    user_input = st.chat_input("Ask a question (you can paste a JD).")
    if user_input or mode == "Summarization":
        if user_input:
            st.session_state["history"].append(("user", user_input))
        query = user_input if user_input else "Summarize the selected dataset’s key ATS and interview guidance."

        with st.spinner("Retrieving and reranking..."):
            docs = retrieve_and_rerank(st.session_state["vectordb"], query, k=top_k, m=top_m)

        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(query, docs, mode=mode, model_name=model_name)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                answer = "Error generating answer. Check API key and logs."

        st.session_state["history"].append(("assistant", answer))

    for role, msg in st.session_state["history"]:
        with st.chat_message(role):
            st.markdown(msg)

    if st.session_state.get("history"):
        last_user = None
        for r, m in reversed(st.session_state["history"]):
            if r == "user":
                last_user = m
                break
        if last_user is None:
            last_user = "Summarize the selected dataset’s key guidance."

        with st.expander("Show source chunks used"):
            docs = retrieve_and_rerank(st.session_state["vectordb"], last_user, k=8, m=3)
            for i, d in enumerate(docs, 1):
                st.markdown(f"**Source {i}** — _{d.metadata.get('dataset','Unknown')}_ — `{d.metadata.get('source','N/A')}`")
                st.code(d.page_content[:1200] + ("..." if len(d.page_content) > 1200 else ""))

if __name__ == "__main__":
    run_app()
