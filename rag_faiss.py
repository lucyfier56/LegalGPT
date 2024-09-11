import os
import logging
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
from transformers import AutoTokenizer
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the tokenizer behavior is explicitly set
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
tokenizer.clean_up_tokenization_spaces = True

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def extract_text_from_page_with_fallback(pdf_reader, page_num, pdf_path):
    """Extract text from a single page. If minimal text is found, fallback to OCR."""
    page = pdf_reader.pages[page_num]
    text = page.extract_text()

    if not text or len(text.strip()) < 20:  # Threshold to determine if OCR is needed
        logging.info(f"Using OCR for page {page_num+1} of {os.path.basename(pdf_path)}")
        images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
        ocr_text = pytesseract.image_to_string(images[0])
        text = f"[OCR]\n{ocr_text}"
    else:
        logging.info(f"Directly extracted text from page {page_num+1} of {os.path.basename(pdf_path)}")
    
    return text

def extract_text_from_hybrid_pdf(pdf_path):
    """Extract text from both selectable and non-selectable regions of the PDF."""
    text = ''
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page_text = extract_text_from_page_with_fallback(pdf_reader, page_num, pdf_path)
                text += f"--- Page {page_num+1} of {os.path.basename(pdf_path)} ---\n{page_text}\n"
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
    return text

def load_pdfs(directory: str) -> List[Document]:
    documents = []
    for file in os.listdir(directory):
        if file.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory, file)
            try:
                extracted_text = extract_text_from_hybrid_pdf(pdf_path)
                documents.append(Document(
                    page_content=extracted_text,
                    metadata={"source": file}
                ))
                logging.info(f"Loaded text from {file}")
            except Exception as e:
                logging.error(f"Error loading {pdf_path}: {e}")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased chunk size for more context
        chunk_overlap=200,  # Increased overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Created {len(split_docs)} document chunks")
    return split_docs

def create_vector_store(documents: List[Document], collection_name: str = 'test') -> FAISS:
    """Create a FAISS vector store for scalable chunk storage."""
    # Initialize FAISS vector store
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )
    
    # Save FAISS index
    vector_store.save_local(f'./faiss_{collection_name}_index')
    
    logging.info(f"FAISS index for '{collection_name}' created with {len(documents)} documents.")
    return vector_store

def setup_rag_components(pdf_directory: str, collection_name: str):
    logging.info("Starting RAG setup process...")

    try:
        documents = load_pdfs(pdf_directory)
        split_docs = split_documents(documents)

        # Process in batches to handle a large number of chunks
        batch_size = 5000
        for i in range(0, len(split_docs), batch_size):
            batch_docs = split_docs[i:i+batch_size]
            vectorstore = create_vector_store(batch_docs, f'{collection_name}_batch_{i//batch_size}')
        
        logging.info("Vector store created and persisted successfully with FAISS")

    except Exception as e:
        logging.error(f"Failed to set up RAG components: {e}")

if __name__ == "__main__":
    pdf_directory = r"/Users/rupinajay/Developer/Legal Rag Gig/HC Orders"
    setup_rag_components(pdf_directory, 'legal_gpt')