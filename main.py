# Import necessary libraries
import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC

# Load environment variables from .env file
load_dotenv()

# Define a function to get the RAG (Retrieval-Augmented Generation) index
@st.cache_resource(show_spinner=False)
def get_rag_index() -> VectorStoreIndex:
    # Initialize Pinecone client
    pc = PineconeGRPC(api_key=os.getenv("PINECONE_API_KEY"))
    # Set the name of the Pinecone index
    index_name = "llamaindex-documentation-helper"
    # Get the Pinecone index
    pinecone_index = pc.Index(index_name)
    # Create a PineconeVectorStore instance
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    # Return a VectorStoreIndex created from the vector store
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Get the RAG index
index = get_rag_index()
# If chat engine doesn't exist in session state, create it
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context", verbose=True
    )

# Configure the Streamlit page
st.set_page_config(
    page_title="LlamaIndex Docs Helper",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Set the title of the Streamlit app
st.title("LlamaIndex Docs Helper 💬🤖")

# Initialize the chat messages if they don't exist in session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me any question about LlamaIndex framework!",
        }
    ]

# Get user input from chat input
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display all messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is not from the assistant, generate a response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get response from the chat engine
            response = st.session_state.chat_engine.chat(message=prompt)
            # Display the response
            st.write(response.response)
            # Extract source nodes from the response
            nodes = [node for node in response.source_nodes]
            # Display each source node in a separate column
            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                with col:
                    st.header(f"Source Node {i+1}: score={node.score}")
                    st.write(node.text)
            # Add the assistant's response to the message history
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
