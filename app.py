import os
import streamlit as st
from typing import List

from openai import OpenAI

from docling.document_converter import DocumentConverter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DOCUMENT_FOLDER = "documents"


st.set_page_config(page_title="Multimodal Agentic RAG System")
st.title("Multimodal Agentic RAG System")


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    client = OpenAI(
        api_key="OPENAI_API_KEY"
    )

    return client


def ingest_documents(folder_path: str) -> List[Document]:

    converter = DocumentConverter()

    docs = []

    for filename in os.listdir(folder_path):

        filepath = os.path.join(folder_path, filename)

        try:
            result = converter.convert(filepath)

            markdown_content = result.document.export_to_markdown()

            docs.append(
                Document(
                    page_content=markdown_content,
                    metadata={
                        "source": filename
                    }
                )
            )

            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Failed processing {filename}: {e}")

    return docs


def chunk_documents(documents: List[Document]):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    return chunks


@st.cache_resource
def build_vectorstore():

    raw_docs = ingest_documents(DOCUMENT_FOLDER)

    chunks = chunk_documents(raw_docs)

    embedding_model = load_embedding_model()

    vectordb = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vectordb


def retrieve_context(query, vectordb, k=5):

    docs = vectordb.similarity_search(query, k=k)

    return docs


def agentic_retrieval(question, vectordb):

    retrieval_queries = [
        question,
        f"Detailed explanation of {question}",
        f"Technical information regarding {question}",
    ]

    collected_docs = []

    for query in retrieval_queries:

        docs = retrieve_context(query, vectordb)

        collected_docs.extend(docs)

    unique_docs = []

    seen = set()

    for doc in collected_docs:

        if doc.page_content not in seen:

            unique_docs.append(doc)

            seen.add(doc.page_content)

    return unique_docs


def generate_answer(question, retrieved_docs, llm_client):

    context = "\n\n".join([
        f"SOURCE: {doc.metadata['source']}\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    prompt = f"""
You are an enterprise AI assistant.

Answer the question ONLY using the provided context.

If information is missing, explicitly say so.

Provide detailed technical explanations.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = llm_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a retrieval grounded enterprise AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


vectordb = build_vectorstore()

llm_client = load_llm()


query = st.text_input("Ask a question")


if st.button("Run Agent"):

    if query.strip() != "":

        with st.spinner("Running agentic retrieval..."):

            retrieved_docs = agentic_retrieval(
                query,
                vectordb
            )

        with st.spinner("Generating answer..."):

            answer = generate_answer(
                query,
                retrieved_docs,
                llm_client
            )

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Retrieved Sources")

        for i, doc in enumerate(retrieved_docs):

            with st.expander(
                f"Document {i+1} - {doc.metadata['source']}"
            ):

                st.write(doc.page_content[:3000])