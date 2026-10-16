import streamlit as st
from pypdf import PdfReader

from rag_vectorstore import build_vector_store
from rag_retrieval import retrieve_relevant_chunks
from rag_llm import load_llm, generate_answer


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)


# ---------- THEME ----------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

with st.sidebar:
    st.markdown("## Display Mode")

    theme = st.radio(
        "Choose theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme_mode == "Dark" else 1,
        horizontal=True,
        key="theme_selector"
    )

    st.session_state.theme_mode = theme


if theme == "Dark":
    bg = "#080B16"
    panel = "#10172A"
    text = "#EDF4FF"
    muted = "#A5B3CC"
    accent = "#00E5FF"
    purple = "#A855F7"
    border = "#263451"
else:
    bg = "#F3F6FF"
    panel = "#FFFFFF"
    text = "#17213B"
    muted = "#64748B"
    accent = "#087CF0"
    purple = "#7C3AED"
    border = "#D9E2F2"


# ---------- CUSTOM CSS ----------
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {bg};
        color: {text};
    }}

    [data-testid="stHeader"] {{
        background: {bg};
    }}

    [data-testid="stSidebar"] {{
        background: {panel};
        border-right: 1px solid {border};
    }}

    [data-testid="stSidebar"] * {{
        color: {text};
    }}

    h1 {{
        color: {accent} !important;
        font-weight: 800 !important;
        text-shadow: 0 0 18px {accent}44;
    }}

    h2, h3 {{
        color: {purple} !important;
    }}

    p, label, li {{
        color: {text};
    }}

    [data-testid="stMetric"] {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 16px;
    }}

    [data-testid="stChatMessage"] {{
        background: {panel};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 12px;
        margin-bottom: 12px;
    }}

    /* PDF uploader styling */
    [data-testid="stFileUploader"] {{
        background: transparent !important;
        border-radius: 14px;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        background: {panel} !important;
        border: 1px dashed {accent} !important;
        border-radius: 14px !important;
    }}

    [data-testid="stFileUploaderDropzone"] * {{
        color: {text} !important;
    }}

    [data-testid="stFileUploaderDropzone"] button {{
        background: {panel} !important;
        color: {accent} !important;
        border: 1px solid {accent} !important;
        border-radius: 9px !important;
    }}

    [data-testid="stFileUploaderDropzone"] button:hover {{
        background: {accent} !important;
        color: {bg} !important;
    }}

    [data-testid="stFileUploaderDropzoneInstructions"] {{
        color: {muted} !important;
    }}

    [data-testid="stFileUploaderFile"] {{
        background: {panel} !important;
        border: 1px solid {border} !important;
        border-radius: 10px !important;
    }}

    [data-testid="stFileUploaderFile"] * {{
        color: {text} !important;
    }}

    .stButton > button {{
        background: linear-gradient(100deg, {accent}, {purple});
        color: #FFFFFF !important;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        transition: 0.2s ease;
    }}

    .stButton > button:hover {{
        box-shadow: 0 0 18px {accent}66;
        transform: translateY(-1px);
    }}

    [data-testid="stChatInput"] {{
        border: 1px solid {accent};
        border-radius: 14px;
        box-shadow: 0 0 12px {accent}22;
    }}

    [data-testid="stExpander"] {{
        border: 1px solid {border};
        border-radius: 12px;
        background: {panel};
    }}

    [data-testid="stCaptionContainer"] p {{
        color: {muted};
    }}

    hr {{
        border-color: {border};
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# ---------- HEADER ----------
st.title("📚 DocuMind AI")

st.markdown(
    f"<p style='color:{muted};font-size:16px;'>"
    "Your PDFs, understood by AI."
    "</p>",
    unsafe_allow_html=True
)


# ---------- SESSION STATE ----------
defaults = {
    "indexed_file": None,
    "pdf_chunks": [],
    "vector_store": None,
    "embedding_model": None,
    "chat_history": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- SIDEBAR UPLOAD ----------
with st.sidebar:
    st.divider()
    st.markdown("## 📄 Your document")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Choose a text-based PDF to get started."
    )

    if st.session_state.vector_store is not None:
        st.success("Document indexed")

        st.caption(
            f"{len(st.session_state.pdf_chunks)} text chunks ready"
        )


# ---------- PDF PROCESSING ----------
if uploaded_pdf:
    file_key = (
        uploaded_pdf.name,
        uploaded_pdf.size
    )

    if st.session_state.indexed_file != file_key:

        # Clear old document state before processing a new PDF.
        st.session_state.indexed_file = None
        st.session_state.vector_store = None
        st.session_state.embedding_model = None
        st.session_state.pdf_chunks = []
        st.session_state.chat_history = []

        try:
            reader = PdfReader(uploaded_pdf)
            chunks = []

            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""

                for i in range(0, len(page_text), 500):
                    part = page_text[i:i + 500]

                    if part.strip():
                        chunks.append({
                            "page": page_num,
                            "text": part
                        })

            if not chunks:
                st.error(
                    "No extractable text found. "
                    "This may be a scanned PDF."
                )

            else:
                with st.spinner(
                    "✨ Reading PDF and creating embeddings..."
                ):
                    embedding_model, client, collection = (
                        build_vector_store(chunks)
                    )

                st.session_state.pdf_chunks = chunks
                st.session_state.embedding_model = embedding_model
                st.session_state.vector_store = collection
                st.session_state.indexed_file = file_key

                st.success("PDF indexed successfully!")

        except Exception as e:
            st.error(f"PDF processing failed: {e}")


# ---------- DOCUMENT DASHBOARD ----------
if st.session_state.vector_store is not None:

    st.markdown("### ⚡ Document overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Text chunks",
        len(st.session_state.pdf_chunks)
    )

    col2.metric(
        "Indexed vectors",
        st.session_state.vector_store.count()
    )

    col3.metric(
        "Pages with text",
        len({
            chunk["page"]
            for chunk in st.session_state.pdf_chunks
        })
    )

    st.divider()

    st.markdown("### 💬 Chat with your PDF")

    st.caption(
        "Ask a question. Answers are generated using retrieved PDF context."
    )

    # Render existing conversation.
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                st.caption(
                    "📚 Sources: " + ", ".join(
                        f"[Page {p}]"
                        for p in message["sources"]
                    )
                )

                with st.expander("View retrieved passages"):
                    for item in message.get("passages", []):
                        st.markdown(f"**Page {item['page']}**")
                        st.write(item["text"])

    question = st.chat_input(
        "Ask anything about your document..."
    )

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner(
                    "🔎 Searching document and generating answer..."
                ):
                    retrieved = retrieve_relevant_chunks(
                        question,
                        st.session_state.embedding_model,
                        st.session_state.vector_store,
                        top_k=5
                    )

                    if not retrieved:
                        answer = (
                            "I couldn't find relevant information "
                            "in this PDF."
                        )
                    else:
                        tokenizer, llm = load_llm()

                        answer = generate_answer(
                            question,
                            retrieved,
                            tokenizer,
                            llm
                        )

                st.markdown(answer)

                source_pages = sorted({
                    item["page"]
                    for item in retrieved
                })

                if source_pages:
                    st.caption(
                        "📚 Sources: " + ", ".join(
                            f"[Page {p}]"
                            for p in source_pages
                        )
                    )

                with st.expander("View retrieved passages"):
                    for item in retrieved:
                        st.markdown(f"**Page {item['page']}**")
                        st.write(item["text"])

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_pages,
                    "passages": retrieved
                })

            except Exception as e:
                st.error(f"Answer generation failed: {e}")

else:
    st.info(
        "👈 Upload a PDF from the sidebar to start chatting."
    )


# ---------- FOOTER ----------
st.markdown(
    f"""
    <div style="
        text-align:center;
        color:{muted};
        padding:28px 0 14px;
        font-size:13px;
        border-top:1px solid {border};
        margin-top:24px;
    ">
        Made with ❤️ by
        <span style="
            color:{accent};
            font-weight:700;
            text-shadow:0 0 10px {accent}44;
        ">Saurabh Gautam</span>
        <br>
        <span style="font-size:11px;">
            DocuMind AI · Powered by embeddings, ChromaDB & Qwen
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
