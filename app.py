import streamlit as st
import pandas as pd

from backend.extractor import extract_text_from_pdf, clean_text, chunk_text
from backend.embedder import embed_texts, semantic_search
from backend.summarizer import summarize_text
from backend.qg import generate_questions_from_text, make_mcq
from backend.mcqs import generate_mcqs_from_text, save_mcqs   # ✅ new MCQ pipeline

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("AI-Powered Study Assistant 📚✨")

# ---------------- File upload ----------------
uploaded = st.file_uploader("Upload PDF / TXT", type=["pdf", "txt"])
if uploaded:
    with open(f"data/uploads/{uploaded.name}", "wb") as f:
        f.write(uploaded.getbuffer())
    st.success("File uploaded.")

    # extract + clean
    raw = extract_text_from_pdf(f"data/uploads/{uploaded.name}")
    raw = clean_text(raw)

    # chunk
    st.info("Extracted text. Chunking...")
    chunks = chunk_text(raw, words_per_chunk=1000)
    st.session_state["chunks"] = chunks     # ✅ save to session state
    st.write(f"Created {len(chunks)} chunks.")

    # embeddings button
    if st.button("Create embeddings (this may take time)"):
        with st.spinner("Embedding..."):
            embeddings = embed_texts([c["text"] for c in chunks])
        st.success("Embeddings ready.")
        st.session_state["embeddings"] = embeddings

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Summaries", "MCQs", "Flashcards", "Chat (semantic)"])

# ---------------- Summaries ----------------
with tab1:
    st.header("Summaries")
    if "chunks" in st.session_state:
        chunks = st.session_state["chunks"]
        summaries = []
        for c in chunks:
            s = summarize_text(c["text"], max_length=120, min_length=40)
            summaries.append({"id": c["id"], "summary": s})

        # show summaries
        for sm in summaries:
            st.write(f"**Chunk {sm['id']}** → {sm['summary']}")
            st.write("---")

# ---------------- MCQs ----------------
with tab2:
    st.header("Generated MCQs")
    if "chunks" not in st.session_state:
        st.info("Upload and process a document first.")
    else:
        chunks = st.session_state["chunks"]

        # quick/full mode toggle
        quick_mode = st.checkbox("Quick mode (use first 5 chunks)", value=True)
        max_chunks = 5 if quick_mode else len(chunks)

        all_mcqs = []
        progress = st.progress(0)
        total = min(max_chunks, len(chunks))

        for i, c in enumerate(chunks[:max_chunks]):
            try:
                mcqs = generate_mcqs_from_text(c["text"], num_q=2)
            except Exception as e:
                st.error(f"MCQ generation failed for chunk {c.get('id', i+1)}: {e}")
                continue

            all_mcqs.extend(mcqs)

            # show MCQs
            for m in mcqs:
                st.write("**Q:**", m["question"])
                for idx, opt in enumerate(m["options"]):
                    st.write(f"{chr(65+idx)}. {opt}")
                st.write("✅ Correct:", m["answer"])
                st.write("---")

            progress.progress((i + 1) / total)

        if all_mcqs and st.button("Save MCQs to CSV"):
            path = save_mcqs(all_mcqs)
            st.success(f"MCQs saved to {path}")

# ---------------- Flashcards ----------------
with tab3:
    st.header("Flashcards")
    if "chunks" in st.session_state:
        chunks = st.session_state["chunks"]
        flashcards = []
        for c in chunks[:20]:
            front = f"Chunk {c['id']} - key idea?"
            back = summarize_text(c["text"], max_length=60, min_length=20)
            flashcards.append({"front": front, "back": back})

        # show simple text flashcards
        for fc in flashcards:
            st.write(f"**Q:** {fc['front']}")
            st.write(f"**A:** {fc['back']}")
            st.write("---")

# ---------------- Semantic Chat ----------------
with tab4:
    st.header("Ask (semantic search)")
    if "chunks" in st.session_state and "embeddings" in st.session_state:
        chunks = st.session_state["chunks"]
        embeddings = st.session_state["embeddings"]

        q = st.text_input("Ask a question about the uploaded document")
        if q and st.button("Search"):
            results = semantic_search(q, [c["text"] for c in chunks], embeddings, top_k=3)
            for r in results:
                st.write(f"Score: {r['score']:.3f}")
                st.write(r["text"][:800] + ("..." if len(r["text"]) > 800 else ""))
                st.write("---")
    else:
        st.info("Upload document and create embeddings to enable chat.")

