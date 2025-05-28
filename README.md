
# Codebase Q\&A with LangChain + Code LLaMA

This project is a **Codebase Question & Answering System** that allows you to ask questions about your codebase using natural language. It leverages **LangChain**, **CodeLLaMA**, **Qdrant**, and **Gradio** to create an interactive chatbot capable of retrieving relevant code snippets and generating answers based on them.

---

## Features

*  Supports `.py`, `.md`, and `.txt` files.
*  Embedding-based similarity search using `BAAI/bge-base-en`.
*  Uses `Qdrant` for vector storage and similarity queries.
*  Code understanding with `codellama/CodeLlama-7b-Instruct-hf`.
*  Simple and intuitive `Gradio` interface for asking code-related questions.

---

## Requirements

Install the required libraries:

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` with the following content:

```txt
gradio
langchain
qdrant-client
transformers
torch
sentence-transformers
```

---

##  How It Works

1. **Convert Code Files to Text**
   Automatically converts all `.py`, `.md`, and `.txt` files in the root-level directory into a text format for processing.

2. **Load & Split Documents**
   Documents are loaded and chunked using LangChain's `RecursiveCharacterTextSplitter`.

3. **Generate Embeddings & Store in Qdrant**
   Embeddings are created using `BAAI/bge-base-en` and stored in a local Qdrant vector store.

4. **Query & Generate Answers**

   * Similar documents are retrieved using similarity search.
   * A prompt is generated using the retrieved context and your question.
   * The answer is generated using `CodeLlama-7b-Instruct-hf`.

5. **Interact via Gradio Interface**
   A web interface is launched where you can type in your question.

---

## Running the App

Ensure Qdrant is running locally:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Then run the script:

```bash
python app.py
```

A Gradio web interface will open in your browser.

---

## Ask Questions Like:

* *What does the main function in this code do?*
* *Where is the database connection configured?*
* *Explain the logic inside the `process_data` function.*

---

## Model Info

* **Embedding Model**: `BAAI/bge-base-en`
* **LLM**: `codellama/CodeLlama-7b-Instruct-hf`

You can switch to other models supported by HuggingFace if needed.

---

## Folder Structure (Flat Layout)

All files are placed in the root of the repo. No subfolders are required.

```
├── app.py
├── README.md
├── requirements.txt
```

---

## Tips

* You can place your code files directly in the repo root (same level as `app.py`).
* On first run, text versions of the code files are auto-generated in a temporary directory used for processing.
* Make sure Qdrant is running before launching the app.

---

