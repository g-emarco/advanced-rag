import os

from dotenv import load_dotenv
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms.ai21 import AI21
from langchain.llms.vertexai import VertexAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.pgvector import PGVector

from ai21.contextual_answers import AI21ContextualAnswers
from queries import query2, query1
from rag.rag import ask_bot_vertex

load_dotenv()

# db = FAISS.load_local(folder_path="faiss_index", embeddings = VertexAIEmbeddings())
db = PGVector.from_existing_index(embedding=VertexAIEmbeddings())
ai21 = AI21(ai21_api_key=os.environ["AI21_API_KEY"])
palm2 = VertexAI()
ai21_contextual_answers = AI21ContextualAnswers(
    ai21_api_key=os.environ["AI21_API_KEY"], model="answer"
)


def ask_bot_ai21(question_query: str) -> str:
    relevant_docs = db.similarity_search(query=question_query, k=5)
    texts = " ,".join([doc.page_content for doc in relevant_docs])

    response = ai21_contextual_answers.invoke(question_query, context=texts)
    return response


def ask_bot2_ai21(question_query: str) -> str:
    prompt = ChatPromptTemplate.from_template(question_query)

    chain = (
        {"context": db.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | ai21_contextual_answers
        | StrOutputParser()
    )
    return chain.invoke(question_query)


if __name__ == "__main__":
    # query = "What if I miss a premium payment?"
    # query = "Who can apply for coverage?"

    query = query2
    print(f"***Question***")
    print(query)
    # answer = ask_bot_ai21(question_query=query)
    answer = ask_bot_vertex(question_query=query)

    print("***answer***:")
    print(answer)
