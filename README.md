# LegalGPT

## Overview

**LegalDocSearchGPT** is an AI-powered Retrieval-Augmented Generation (RAG) system designed for analyzing and querying legal PDF documents. It uses advanced machine learning models to extract, index, and retrieve relevant information from structured and unstructured legal content. The system integrates FAISS for vector storage and Groq's API for generating accurate, context-based responses.

## Features

- **PDF Text Extraction**: Extracts text from legal PDFs, including non-selectable content via OCR.
- **Chunk Splitting**: Splits large legal documents into manageable chunks for better context understanding.
- **FAISS Indexing**: Efficiently indexes document chunks for fast and accurate information retrieval.
- **AI-Powered Querying**: Uses a GPT-based model to generate insightful legal responses based on document content.
- **Streamlit UI**: Provides an interactive web interface for querying and managing legal conversations.

## Tech Stack

- **LangChain**: For document processing and chaining LLM-based models.
- **FAISS**: Vector database for similarity search and retrieval.
- **Groq API**: Handles text generation tasks.
- **Streamlit**: Interactive UI for users to query legal documents.
- **OCR with PyTesseract**: Extracts text from non-selectable legal PDFs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lucyfier56/LegalGPT.git
   cd LegalGPT

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

## Usage
1. Index Documents
   Place your PDF documents in a directory and run the rag_faiss.py script to index them:

      ```bash
     python rag_faiss.py
  - This will process the PDFs, extract the text, and create a FAISS vector store for document retrieval.

2. Run the Streamlit App
   Use the Streamlit UI for querying and interacting with the system:

   ```bash
   streamlit run user_groq_streamlit.py
   
3. Chat with the Assistant
Ask legal queries, and the assistant will retrieve relevant context and generate responses.

## Team Members

This project was developed by the following team members:

- **Rudra Panda** - [GitHub Profile](https://github.com/lucyfier56)
- **Rupin Ajay** - [GitHub Profile](https://github.com/rupinajay)
- **Sanjay P N** -[GitHub Profile](https://github.com/sanjayperam04)
- **Sukrith PVS** -[GitHub Profile](https://github.com/sukrithpvs)

We collaborated to build this project, each contributing to different aspects of the application.
