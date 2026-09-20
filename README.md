# Multi-Modal RAG

A local Retrieval-Augmented Generation (RAG) system built with **LangChain, ChromaDB, Hugging Face embeddings, and Ollama**.

The project processes documents, stores their embeddings in a vector database, retrieves the most relevant document chunks for a user query, and generates answers using a locally running LLM.

## Features

* 📄 Document-based question answering
* 🔎 cosine similarity search using ChromaDB
* 🧠 `BAAI/bge-small-en-v1.5` Hugging Face embeddings
* 🤖 Local LLM inference using Ollama
* 💬 Conversational chat history
* 🔐 Runs locally without sending document content to a cloud LLM API
* 🧩 Modular multi-file project structure
* 🎯 Retrieves the top 3 relevant document chunks for each query
* 🚫 Instructs the LLM to answer only from retrieved document context

## Tech Stack

* **Python**
* **LangChain**
* **LangGraph**
* **ChromaDB**
* **Hugging Face**
* **Ollama**
* **Llama 3.2 3B**
* **BGE-small-en-v1.5**
* **PyMuPDF**


## Project Structure

```text
multi_modal_RAG/
│
├── .venv/
│── extracted_images
├── chroma_db/
│
├── embedding.py
├── retriever.py
├── chunks.py
├── ingestion.py
|
├── .gitignore
└── sample.pdf
```

> `.venv/` and `chroma_db/` are local/generated directories and should not be committed to GitHub.

## How It Works

The basic RAG pipeline is:

```text
                Documents
                    │
                    ▼
             Document Processing
                    │
                    ▼
              Text Chunking
                    │
                    ▼
          Hugging Face Embeddings
          BAAI/bge-small-en-v1.5
                    │
                    ▼
               ChromaDB
                    │
                    │
User Query ─────────┤
                    ▼
              cosine Similarity
                    │
              Top 3 Documents
                    │
                    ▼
            Retrieved Context
                    │
                    ▼
             Ollama Llama 3.2
                    │
                    ▼
                 Answer
```

## Retriever

The main retrieval and generation logic is implemented in `retriever.py`.

### Embeddings

The project uses:

```python
HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
```

These embeddings convert document chunks and user queries into vectors that can be compared using semantic similarity.

### Vector Database

ChromaDB is configured with cosine similarity:

```python
database = Chroma(
    embedding_function=embedding,
    persist_directory="chroma_db",
    collection_metadata={"hnsw:space": "cosine"}
)
```

The retriever returns the top 3 relevant chunks:

```python
retriever = database.as_retriever(
    search_kwargs={"k": 3}
)
```

### Local LLM

The project uses Ollama with:

```python
ChatOllama(
    model="llama3.2:3b",
    temperature=0.2
)
```

The model runs locally through Ollama.

## Conversational RAG

The application maintains chat history using LangChain message objects:

```python
chat_history = []
```

For every query:

1. The current query is used for retrieval.
2. The top relevant document chunks are retrieved.
3. Retrieved context is added to the system message.
4. Previous conversation history is included.
5. The LLM generates the answer.
6. The current question and answer are added to chat history.

Importantly, **retrieval is performed using only the current user query**, while previous messages are provided to the LLM as conversational context.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/utkarshsingh171/Multi_Modal_RAG.git
cd multi_modal_RAG
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install langchain langchain-community langchain-classic langgraph langgraph-prebuilt chromadb langchain-chroma langchain-ollama langchain-huggingface sentence-transformers
```

## Ollama Setup

Install Ollama and make sure it is running.

Pull the Llama model:

```powershell
ollama pull llama3.2:3b
```

The embedding model is downloaded through Hugging Face when the application initializes:

```text
BAAI/bge-small-en-v1.5
```

## Running the Project

After installing the dependencies and creating the vector database, run:

```powershell
python retriever.py
```

You should see:

```text
Ask your question. Type 'quit' to exit.
```

Then enter a question:

```text
Query: What is this document about?
```

The system retrieves relevant chunks and generates an answer using the local LLM.

To exit:

```text
quit
```

## Example

```text
Ask your question. Type 'quit' to exit.

Query: What is the main topic of the document?

Assistant:
The document discusses ...
```

## RAG Pipeline Details

### Retrieval

For every query:

```python
relevant_docs = retriever.invoke(query)
```

The system retrieves the three most relevant document chunks.

### Context Construction

The retrieved chunks are combined:

```python
context_text = "\n\n".join(
    [doc.page_content for doc in relevant_docs]
)
```

### Grounded Generation

The LLM receives instructions to answer using only the retrieved context:

```text
Answer the user's question using ONLY the provided document context.
If the answer cannot be found in the context, state that you don't know.
```

This helps reduce answers that are unrelated to the indexed documents.

## Why Local RAG?

This project uses local models through Ollama instead of relying on a hosted LLM API.

Advantages include:

* Local document processing
* No API key required for LLM inference
* Better control over private documents
* Ability to experiment with different local models
* No per-request LLM API cost

## Current Limitations

* Retrieval currently uses only the current user query.
* The system retrieves the top 3 chunks.
* Answer quality depends on document chunking and embedding quality.
* `llama3.2:3b` is a relatively small local model.
* The current retriever is primarily text-based.
* Large document collections may require additional retrieval optimization.

## Future Improvements

* [ ] Multimodal image retrieval
* [ ] Vision-language model integration
* [ ] PDF table extraction
* [ ] Better document chunking
* [ ] Hybrid search
* [ ] Reranking retrieved documents
* [ ] Metadata filtering
* [ ] Query rewriting
* [ ] Conversational query rewriting
* [ ] Streaming responses
* [ ] Web-based chat interface
* [ ] Evaluation using RAG metrics
* [ ] LangGraph-based agentic retrieval workflow

## License

This project is intended for learning, experimentation, and research purposes.
