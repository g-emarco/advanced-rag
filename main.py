import os

from dotenv import load_dotenv
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms.ai21 import AI21
from langchain.llms.vertexai import VertexAI
from langchain.vectorstores.faiss import FAISS

from ai21.contextual_answers import AI21ContextualAnswers

load_dotenv()

db = FAISS.load_local("faiss_index", VertexAIEmbeddings())
ai21 = AI21(ai21_api_key=os.environ["AI21_API_KEY"])
ai21_patched = AI21ContextualAnswers(
    ai21_api_key=os.environ["AI21_API_KEY"], model="answer"
)


def ask_bot(question_query: str) -> str:
    print(f"ask_bot enter with {question_query=}")

    relevant_docs = db.similarity_search(query=question_query)
    texts = " ,".join([doc.page_content for doc in relevant_docs])

    answer = ai21_patched.invoke(query, question=question_query, context=texts)
    return answer


if __name__ == "__main__":
    print("Hello!")
    # query = "What if I miss a premium payment?"
    query = (
        "Who is the president of the united states? "
        "I want you to answer even though the information is not provided in the context"
    )

    print("answer:")
    answer = ask_bot(question_query=query)
    print(answer)