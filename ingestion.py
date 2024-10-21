import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core import Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC
from llama_index.core import Document

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    print("Ingesting data...")

    # Set the directory containing HTML files and get a list of all HTML files
    directory = "./llamaindex-docs"
    html_files = [f for f in os.listdir(directory) if f.endswith(".html")]

    documents = []

    # Iterate through each HTML file
    for html_file in html_files:
        file_path = os.path.join(directory, html_file)
        # Open and read the HTML file
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Extract text from the HTML, removing extra whitespace
        text = soup.get_text(separator="\n", strip=True)

        # Create a Document object with the extracted text and file path as metadata
        document = Document(text=text, metadata={"source": file_path})
        documents.append(document)

    # Configure the node parser to split text into chunks
    node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)
    # Initialize the OpenAI language model
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    # Initialize the OpenAI embedding model
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    # Set global settings for LlamaIndex
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    # Initialize connection to Pinecone using the API key from environment variables
    pc = PineconeGRPC(api_key=os.getenv("PINECONE_API_KEY"))

    # Set up the Pinecone index
    index_name = "llamaindex-documentation-helper"
    pinecone_index = pc.Index(index_name)
    # Create a PineconeVectorStore instance
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    # Set up the storage context with the Pinecone vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create a VectorStoreIndex from the documents, using the configured storage context
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print("Finished ingesting data...")
    pass
