import os
import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def convert_files_to_txt(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(('.py', '.md', '.txt')):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, src_dir)
                new_path = os.path.join(dst_dir, rel_path)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                with open(new_path, 'w', encoding='utf-8') as f:
                    f.write(content)

convert_files_to_txt('codebase', 'text_codebase')


loader = DirectoryLoader('text_codebase', loader_cls=TextLoader)
docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())


embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en")
qdrant = Qdrant.from_documents(
    docs,
    embedding_model,
    url="http://localhost:6333",
    prefer_grpc=False,
    collection_name="codebase"
)


model_name = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

def query_codebase(q):
    results = qdrant.similarity_search(q, k=5)
    context = "\n".join([doc.page_content for doc in results])
    prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
    return llm(prompt)

gr.Interface(fn=query_codebase, inputs="text", outputs="text", title="Codebase Q&A").launch()
