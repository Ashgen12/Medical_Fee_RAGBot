# RAG-Based Fee Schedule Analysis Chatbot  

![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)  

💡 A sophisticated, end-to-end **Retrieval-Augmented Generation (RAG)** system designed to perform complex analysis and calculations on unstructured PDF fee schedules.  
📑 This project transforms raw documents, including scanned and table-heavy formats, into an interactive and queryable knowledge base.  

---

## 📘 Abstract  
This project implements a complete **RAG pipeline** to address the challenge of extracting and reasoning over information locked within complex PDF documents.  

✨ The system integrates:  
- 📝 A robust data ingestion engine featuring **OCR** for scanned pages and advanced table extraction.  
- ⚡ A quantized Small Language Model (`Phi-3-mini`) to ensure high-fidelity responses while maintaining computational efficiency.  
- 🔄 A file monitoring system that automatically updates the vector knowledge base in response to changes in the source documents.  

---

## 🏗️ System Architecture  

The application is architected around two primary workflows:  

### 🔹 Ingestion Pipeline (Offline)  
📄 PDF Documents  
   ⬇️ Content Extractor  
   - 📝 Text → Chunker  
   - 🖼️ Image-based Page → OCR (Tesseract) → Chunker  
   - 📊 Tables → Table-to-Markdown → Chunker  
   ⬇️ Embedding Model  
   ⬇️ 🗄️ ChromaDB Vector Store  

### 🔹 Inference Pipeline (Online)  
💬 User Query  
   ⬇️ Embedding Model  
   ⬇️ 🔎 Similarity Search (from ChromaDB)  
   ⬇️ Retrieve Top-k Chunks  
   ⬇️ Prompt Augmentation (merging query + context)  
   ⬇️ 🤖 Quantized LLM (Phi-3)  
   ⬇️ ✅ Generated Response  

---

## ⚙️ Core Components & Technologies  

### 📂 Data Ingestion & Preprocessing  
- **PDF Parsing (PyMuPDF):** High-performance text and native table extraction.  
- **Optical Character Recognition (Pytesseract):** Fallback for scanned PDFs.  
- **Semantic Chunking (LangChain):** RecursiveCharacterTextSplitter for optimized retrieval.  

### 🔍 Vectorization & Retrieval  
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` ⚡  
- **Vector Store:** ChromaDB 🗄️ (local-first, persistent).  

### 🤖 Generative Core  
- **Language Model:** `microsoft/Phi-3-mini-4k-instruct`  
- **Quantization:** `bitsandbytes` (4-bit NF4).  

### 🌐 Application Layer  
- **Web Interface (Gradio):** 🖥️ Interactive chat UI.  

---

## 📑 Algorithmic Details  

### 1️⃣ Dynamic Ingestion Pipeline  
- 🔐 **File Monitoring:** SHA256 hash of PDFs stored in `file_hashes.json`.  
- 🆕 **Delta Detection:** Detects new, modified, and deleted files.  
- ♻️ **Atomic Updates:** Keeps ChromaDB in sync with documents.  

### 2️⃣ Inference & Prompting Strategy  
- 🧭 **Query Embedding:** Vectorized using the same embedding model.  
- 🔎 **Context Retrieval:** Top-5 chunks from ChromaDB.  
- 📝 **Prompt Engineering:** Injects chunks into a structured prompt.  
- 🚫 **Constrained Generation:** Grounded responses, reduced hallucination.  

---

## ⚡ Setup and Execution  

### 🔧 Prerequisites  
- Python 3.10+ 🐍  
- **Tesseract OCR Engine** (installed & in PATH) 🔤  
- NVIDIA GPU with CUDA (recommended) 🎮  

### 📥 Installation  

```bash
# Clone & Enter Repository
git clone <repository_url>
cd <repository_name>

# Setup Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### ▶️ Execution
📂 Add your PDF files into the /documents directory.
🚀 Run the application:

```bash
python app.py
```

The server will start at 👉 http://127.0.0.1:7860.
(first run may take a few minutes for downloads & indexing)

## 📊 Performance & Scalability

### Performance:
⚡ 4-bit quantization enables GPU-friendly operation (RTX GPUs ≥8GB VRAM).
🐢 CPU-only inference is very slow.

### Scalability:
🧑‍💻 Local ChromaDB = great for single/small users.
🌍 For larger apps: move to Pinecone, Weaviate, or managed ChromaDB.

## 📜 License
This project is licensed under the MIT License.
See the LICENSE file for details.
