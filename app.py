import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import pandas as pd
from pathlib import Path
import logging
import hashlib
import json
import time
import os

# --- Configuration ---
# Set up logging to monitor the process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- TESSERACT OCR CONFIGURATION (MODIFY IF NEEDED) ---
# If tesseract is not in your PATH, uncomment and set the path here
# For Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define paths for documents and the persistent vector database
DOCUMENTS_PATH = Path("documents")
CHROMA_DB_PATH = Path("chroma_db")
FILE_HASH_TRACKER = Path("file_hashes.json")

# Hugging Face model identifiers
LLM_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "fee_schedule_docs"

# --- 1. Robust Data Loading and Processing with OCR ---

def get_file_hash(filepath):
    """Calculates the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_file_hashes():
    """Loads the dictionary of processed file hashes."""
    if FILE_HASH_TRACKER.exists():
        with open(FILE_HASH_TRACKER, 'r') as f:
            return json.load(f)
    return {}

def save_file_hashes(hashes):
    """Saves the dictionary of processed file hashes."""
    with open(FILE_HASH_TRACKER, 'w') as f:
        json.dump(hashes, f, indent=4)

def extract_data_from_pdf(pdf_path: Path) -> str:
    """
    Extracts text and tables from a PDF, using OCR for image-based pages.
    Tables are converted to Markdown format.
    """
    full_content = ""
    logging.info(f"Opening PDF: {pdf_path.name}")
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        # 1. Extract text
        text = page.get_text()

        # 2. If text is minimal, assume it's an image and perform OCR
        if len(text.strip()) < 100:
            logging.warning(f"Page {page_num + 1} of {pdf_path.name} has little text. Attempting OCR.")
            try:
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                ocr_text = pytesseract.image_to_string(image)
                text = ocr_text
                logging.info(f"Successfully extracted text from page {page_num + 1} via OCR.")
            except Exception as e:
                logging.error(f"OCR failed for page {page_num + 1} of {pdf_path.name}: {e}")

        full_content += text + "\n"

        # 3. Extract tables using PyMuPDF and convert to Markdown
        tables = page.find_tables()
        # The TableFinder object `tables` has no len(). We must check and get len from its `.tables` attribute.
        if tables.tables:
            logging.info(f"Found {len(tables.tables)} table(s) on page {page_num + 1}.")
            for table in tables:
                df = table.to_pandas()
                # Clean up empty dataframes
                if not df.empty:
                    # Promote first row to header if it looks like one
                    if all(isinstance(c, str) for c in df.iloc[0]) and not any(df.iloc[0].isnull()):
                        df.columns = df.iloc[0]
                        df = df[1:]
                    table_markdown = "\n" + df.to_markdown(index=False) + "\n"
                    full_content += f"\n--- TABLE DATA ---\n{table_markdown}\n--- END TABLE DATA ---\n"

    doc.close()
    return full_content


def process_and_chunk_documents():
    """
    Processes all PDFs in the DOCUMENTS_PATH, handling updates intelligently.
    Returns a list of chunks to be added and a list of sources to be deleted.
    """
    if not DOCUMENTS_PATH.exists():
        DOCUMENTS_PATH.mkdir()
        logging.warning("Created 'documents' directory. Please add your PDFs and restart.")
        return [], []

    processed_hashes = load_file_hashes()
    current_files = {p.name: p for p in DOCUMENTS_PATH.glob("*.pdf")}
    current_hashes = {name: get_file_hash(path) for name, path in current_files.items()}

    new_or_updated_files = []
    processed_filenames = list(processed_hashes.keys())

    # Identify new or updated files
    for name, hash_val in current_hashes.items():
        if name not in processed_hashes or processed_hashes[name] != hash_val:
            new_or_updated_files.append(name)
            logging.info(f"Detected new/updated file: {name}")

    # Identify deleted files
    deleted_files = [name for name in processed_filenames if name not in current_files]
    if deleted_files:
        logging.info(f"Detected deleted files: {', '.join(deleted_files)}")

    all_chunks_to_add = []
    if new_or_updated_files:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        for filename in new_or_updated_files:
            pdf_path = current_files[filename]
            content = extract_data_from_pdf(pdf_path)
            if content:
                chunks = text_splitter.create_documents([content])
                for chunk in chunks:
                    chunk.metadata = {"source": filename}
                all_chunks_to_add.extend(chunks)
                logging.info(f"Split {filename} into {len(chunks)} chunks.")

    # Update the hash tracker
    for filename in deleted_files:
        del processed_hashes[filename]
    for filename in new_or_updated_files:
        processed_hashes[filename] = current_hashes[filename]
    save_file_hashes(processed_hashes)

    return all_chunks_to_add, deleted_files


# --- 2. Vector Database Setup ---

def setup_vector_database(chunks_to_add, sources_to_delete):
    """
    Initializes ChromaDB, and updates it with new/modified/deleted document chunks.
    """
    logging.info("Setting up vector database...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    # Delete documents from deleted source files
    if sources_to_delete:
        for source in sources_to_delete:
            logging.info(f"Deleting documents from source: {source}")
            collection.delete(where={"source": source})

    # Delete documents from updated source files before re-adding
    updated_sources = list(set(chunk.metadata['source'] for chunk in chunks_to_add))
    if updated_sources:
        for source in updated_sources:
             logging.info(f"Deleting old documents from updated source: {source}")
             collection.delete(where={"source": source})

    # Add new chunks
    if chunks_to_add:
        logging.info(f"Adding {len(chunks_to_add)} new chunks to the database.")
        documents = [chunk.page_content for chunk in chunks_to_add]
        metadatas = [chunk.metadata for chunk in chunks_to_add]
        # Create unique IDs based on content and source
        ids = [hashlib.sha256(f"{chunk.metadata['source']}-{i}".encode()).hexdigest() for i, chunk in enumerate(chunks_to_add)]
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            logging.info(f"Added batch {i//batch_size + 1} to ChromaDB.")

    logging.info(f"Vector database setup complete. Collection '{CHROMA_COLLECTION_NAME}' contains {collection.count()} documents.")
    return collection


# --- 3. Large Language Model (LLM) Setup ---
from transformers import BitsAndBytesConfig # <-- Add this import

def load_llm_pipeline():
    """Loads the Phi-3 model and tokenizer with 4-bit quantization."""
    logging.info("Loading the LLM with 4-bit quantization. This may take a moment...")
    
    # Configuration for 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    try:
        # Note: torch_dtype is now handled by the quantization config
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=quantization_config, # <-- Add this
            trust_remote_code=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        
        # When using quantization, it's good practice to set do_sample=True
        llm_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_new_tokens=1024, 
            temperature=0.1,
            do_sample=True # <-- Add this to resolve the warning
        )
        logging.info("LLM pipeline loaded successfully.")
        return llm_pipeline
    except Exception as e:
        logging.error(f"Failed to load LLM: {e}")
        logging.error("If you are on a system without a powerful GPU, the model might be too large. Try a smaller model if issues persist.")
        return None

# --- 4. Chatbot Logic (RAG Core) ---

def format_prompt(query, context_chunks):
    """Creates a detailed and structured prompt for the LLM."""
    context_str = "\n\n---\n\n".join(context_chunks)
    prompt = f"""<|system|>
You are an expert financial assistant specializing in fee schedule calculations. Your task is to provide a clear, accurate, and complete answer based ONLY on the provided context.

**Instructions:**
1.  **Identify all relevant values:** Find the procedure codes, relative value units (RVUs), and conversion factors from the context.
2.  **Show your work:** For each procedure, clearly write down the formula used (e.g., RVU x Conversion Factor = Fee) and the calculated amount.
3.  **Calculate the final total:** If the user asks for a total, you MUST explicitly sum the individual fees to get a final number.
4.  **Provide a summary:** End your response with a bolded, clear summary statement, like "**The total fee for procedures X and Y is $Z.**"
5.  **Be precise:** Do not use external knowledge. If the information is not in the context, state that clearly.

**CONTEXT FROM DOCUMENTS:**
{context_str}<|end|>
<|user|>
{query}<|end|>
<|assistant|>
"""
    return prompt

def get_chatbot_response(query, collection, llm_pipe, embedding_model):
    """The main RAG function."""
    if not query:
        return "Please ask a question."
    if collection.count() == 0:
        return "The document database is empty. Please add PDF files to the 'documents' folder and restart."

    # 1. Embed the query
    query_embedding = embedding_model.encode(query)

    # 2. Retrieve relevant context from ChromaDB
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=5)
    retrieved_documents = results['documents'][0]

    # 3. Format the prompt and generate response
    final_prompt = format_prompt(query, retrieved_documents)
    logging.info("Generating response from LLM...")
    response_data = llm_pipe(final_prompt)
    full_response = response_data[0]['generated_text']
    assistant_response = full_response.split("<|assistant|>")[-1].strip()

    return assistant_response

# --- 5. Main Execution and Gradio UI ---

if __name__ == "__main__":
    start_time = time.time()
    
    # Process documents and update the database
    chunks_to_add, sources_to_delete = process_and_chunk_documents()
    db_collection = setup_vector_database(chunks_to_add, sources_to_delete)

    # Load the models
    text_embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)
    llm_pipeline_main = load_llm_pipeline()

    setup_duration = time.time() - start_time
    logging.info(f"--- System is ready. Setup took {setup_duration:.2f} seconds. ---")

    if db_collection is None or llm_pipeline_main is None:
        logging.error("Failed to initialize the system. Please check the logs.")
    else:
        def chat_interface_fn(message, history):
            return get_chatbot_response(message, db_collection, llm_pipeline_main, text_embedding_model)

        gr.ChatInterface(
            fn=chat_interface_fn,
            title="Fee Schedule Chatbot 📄",
            description="Ask me anything about the fee schedules in the provided PDFs. I can find information and perform calculations for you.",
            examples=[
                "What is the ground rule for anesthesia time?",
                "What is the fee for procedure code 99203?",
                "Calculate the total fee for procedures 99214 and 99215.",
            ],
            theme="soft"
        ).launch()
