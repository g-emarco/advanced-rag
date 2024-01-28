import os
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()
from langchain.embeddings import VertexAIEmbeddings

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.pgvector import PGVector

from langchain_core.documents import Document


def split_with_ai21_segmentation(data: str) -> List[str]:
    url = "https://api.ai21.com/studio/v1/segmentation"

    payload = {"sourceType": "TEXT", "source": data}
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.environ['AI21_API_KEY']}",
    }

    response = requests.post(url, json=payload, headers=headers)
    response = response.json()

    return response["segments"]


def main():
    print("Going to ingest aspirin drug information...")
    loader = PyPDFLoader("aspirin/Aspirin-Protect-en.pdf")
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
    regular_splits = splitter.split_documents(documents=raw_documents)
    print(f"splitted for {len(regular_splits)} splits")

    ai21_segmentations = split_with_ai21_segmentation(
        data=raw_documents[0].page_content
    )
    ai_21_segmentations_documents = [
        Document(page_content=segment["segmentText"]) for segment in ai21_segmentations
    ]
    print(f"segmented for {len(ai21_segmentations)} segments")

    db = PGVector.from_documents(
        documents=regular_splits,
        embedding=embeddings,
        connection_string=os.environ["CONNECTION_STRING_MED_REGULAR"],
    )

    print("finished ingesting...")


if __name__ == "__main__":
    main()
