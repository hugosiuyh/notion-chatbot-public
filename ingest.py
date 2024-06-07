import os
import openai
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import streamlit as st

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Define function to add metadata
def add_metadata(documents, folder):
    for doc in documents:
        doc.metadata["folder"] = folder
        doc.metadata["filename"] = Path(doc.metadata["source"]).name
    return documents

# Load the notion content located in the notion_content folder
notion_content_folder = "notion_content"
loader = NotionDirectoryLoader(notion_content_folder)
documents = loader.load()

# Add metadata to documents
documents = add_metadata(documents, notion_content_folder)

# Split Notion content into smaller chunks
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["#", "##", "###", "\n\n", "\n", "."],
    chunk_size=1500,
    chunk_overlap=100)
docs = markdown_splitter.split_documents(documents)

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings()

# Convert all chunks into vector embeddings using OpenAI embedding model
# Store all vectors in FAISS index and save locally to 'faiss_index'
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

print('Local FAISS index has been successfully saved.')
