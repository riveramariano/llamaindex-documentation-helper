# LlamaIndex Documentation Helper

**_Notes: You need to have python and pipenv installed._**

## Functionality

This project is a simple RAG implementation that allows you to ask questions about the LlamaIndex documentation and get answers based on the documentation.

### Files Description

- download_docs.py: This file downloads the LlamaIndex documentation and saves it in a local folder.
- ingest.py: This file is used to ingest the LlamaIndex documentation into a Pinecone index.
- main.py: This file contains the RAG implementation.

### Run Locally

- Clone the repository `https://github.com/riveramariano/llmaindex-documentation-helper.git`
- Run `pipenv install & pipenv shell` to initialize the virtual env:
- Run `streamlit run main.py` to start the application
