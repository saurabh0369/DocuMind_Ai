# 📚 DocuMind AI — PDF Intelligence Chatbot

An AI-powered PDF question-answering application built using **Retrieval-Augmented Generation (RAG)**. Upload a PDF, ask questions in natural language, and receive context-aware answers with page-level source references.

🔗 **Live Demo:** [Try DocuMind AI](https://viewable-heaviness-twisted.ngrok-free.dev)

## ✨ Features

- **PDF Text Extraction:** Extract text and preserve page metadata using PyPDF.
- **Text Chunking:** Split extracted content into smaller chunks for efficient retrieval.
- **Semantic Search:** Generate text embeddings using Sentence Transformers.
- **Vector Database:** Store and retrieve document chunks using ChromaDB.
- **Hybrid Retrieval:** Combine semantic and keyword-based retrieval to find relevant context.
- **LLM-Powered Answers:** Generate contextual responses using Qwen2.5-3B-Instruct with 4-bit quantization.
- **Page-Level Citations:** Display source page references alongside answers.
- **Interactive Interface:** Use a Streamlit application with Dark and Light themes.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| PDF Processing | PyPDF |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| Language Model | Qwen2.5-3B-Instruct |
| Model Framework | Hugging Face Transformers |
| Quantization | BitsAndBytes (4-bit) |
| Deployment Demo | Google Colab + ngrok |

## 🏗️ Architecture

```text
PDF Upload
    ↓
Text Extraction and Page Metadata
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
ChromaDB Vector Store
    ↓
Hybrid Retrieval
    ↓
Relevant Context
    ↓
Qwen2.5-3B-Instruct
    ↓
Answer with Page References
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- A CUDA-enabled GPU for the configured 4-bit Qwen inference
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/saurabh0369/DocuMind_Ai.git
cd DocuMind_Ai
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

**Note:** The application uses a quantized Qwen model and may require a compatible GPU and additional model memory. The first model load may take time to download the required files.

## 📖 Usage

1. Launch the Streamlit application.
2. Upload a text-based PDF.
3. Wait for the document to be processed and indexed.
4. Ask questions about the uploaded document.
5. Review the generated answer and its cited pages.
6. Expand the retrieved passages to inspect the supporting text.

## 📁 Project Structure

```text
DocuMind_Ai/
├── DocuMind_AI.ipynb
├── app.py
├── rag_vectorstore.py
├── rag_retrieval.py
├── rag_llm.py
├── requirements.txt
└── README.md
```

## ⚠️ Limitations

- Scanned PDFs may require OCR before their text can be extracted.
- Answer quality depends on document quality and retrieval relevance.
- Model inference can be slow on limited hardware.
- The public demo depends on the availability of its hosted runtime and tunnel.

## 👨‍💻 Author

**Saurabh Gautam**

[GitHub Profile](https://github.com/saurabh0369)

---

*Your PDFs, understood by AI.*
