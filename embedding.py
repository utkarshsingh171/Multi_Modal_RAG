from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chunks import chunks

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
)

print("stored",len(chunks),"chunks")




