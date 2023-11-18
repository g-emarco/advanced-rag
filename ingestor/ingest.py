from dotenv import load_dotenv
from langchain.embeddings import VertexAIEmbeddings
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.vectorstores.faiss import FAISS

load_dotenv()

if __name__ == "__main__":
    print("Going to ingest pinecone documentation...")
    loader = UnstructuredHTMLLoader("../lemonade/faq.html")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    embeddings = VertexAIEmbeddings()
    splits = splitter.split_documents(documents=raw_documents)

    db = FAISS.from_documents(splits, embeddings)
    db.save_local("faiss_index")
    print("finished ingesting...")
