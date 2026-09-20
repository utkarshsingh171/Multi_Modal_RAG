from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion import texts

all_text = "\n".join(item["text"] for item in texts)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap = 200
)

chunks = splitter.split_text(all_text)

print(len(chunks))