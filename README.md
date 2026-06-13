# Multimodal Agentic RAG System

## Introduction
An AI-powered document question-answering application that utilizes a Retrieval-Augmented Generation (RAG) architecture. By combining document ingestion, semantic search, and large language models, this system provides highly accurate, context-aware answers that are strictly grounded in the user's uploaded knowledge base.

## Technologies Used
*   **Language:** Python
*   **Interface:** Streamlit
*   **AI & Frameworks:** LangChain, OpenAI GPT-4.1 Mini
*   **Embeddings & Search:** Hugging Face Embeddings (`sentence-transformers/all-MiniLM-L6-v2`), FAISS Vector Database
*   **Document Processing:** Docling, RecursiveCharacterTextSplitter

## How It Works
1.  **Document Ingestion:** Documents placed in the `documents/` directory are processed and converted into a clean markdown format using Docling.
2.  **Chunking:** The extracted text is divided into overlapping chunks using `RecursiveCharacterTextSplitter` to preserve semantic context.
3.  **Vectorization:** Each text chunk is converted into vector embeddings using the Hugging Face MiniLM sentence transformer model.
4.  **Storage:** The embeddings are indexed and stored in a FAISS vector database for rapid semantic similarity search.
5.  **Agentic Retrieval:** When a user queries the system, it generates multiple search query reformulations to ensure comprehensive context retrieval.
6.  **Answer Generation:** Relevant chunks are retrieved, duplicates are filtered out, and the consolidated context is fed to GPT-4.1 Mini. The LLM generates an answer strictly based on the retrieved data.
7.  **Source Transparency:** The Streamlit UI displays the generated answer alongside the specific source documents and chunks used to formulate it.

## Features
*   **Advanced RAG Architecture:** Context-grounded answer generation to eliminate hallucinations.
*   **Multimodal Ingestion:** Robust document conversion and processing using Docling.
*   **Agentic Search:** Improves accuracy by reformulating user queries to search the vector space from multiple angles.
*   **Efficient Processing:** Smart document chunking with overlap preservation and duplicate context elimination.
*   **High Performance:** Cached embedding models and vector stores for rapid retrieval.
*   **Interactive UI:** User friendly Streamlit interface featuring full source transparency for every generated answer.

## Demo

https://github.com/user-attachments/assets/7c9d065e-c73e-4613-9508-edac7373b6f0

