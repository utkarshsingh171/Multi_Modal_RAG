from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.messages import SystemMessage,AIMessage,HumanMessage

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

database =Chroma(
    embedding_function=embedding,
    persist_directory="chroma_db",
    collection_metadata={"hnsw:space":"cosine"}
) 

model = ChatOllama(model="llama3.2:3b",temperature=0.2)

retriever = database.as_retriever(search_kwargs={"k":3})

chat_history = []

def get_answer(query: str) -> str:
    # 1. Retrieve documents using ONLY the current user query
    relevant_docs = retriever.invoke(query)
    
    # Format retrieved document text
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 2. Construct System Prompt with retrieved context
    system_prompt = SystemMessage(
        content=(
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided document context. If the answer cannot be found in the context, "
            "state that you don't know.\n\n"
            f"Context:\n{context_text}"
        )
    )
    
    # 3. Assemble full payload: System message + Chat history + Current user query
    messages = [system_prompt] + chat_history + [HumanMessage(content=query)]
    
    # 4. Generate response
    result = model.invoke(messages)
    content = result.content
    
    # 5. Update history in correct chronological order
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=content))
    
    return content

def main():
    print("Ask your question. Type 'quit' to exit.")
    while True:
        query = input("\nQuery: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break
            
        answer = get_answer(query)
        print(f"\nAssistant:\n{answer}")


if __name__ == "__main__":
    main()