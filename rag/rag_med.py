import os

from langchain_community.embeddings import VertexAIEmbeddings
from langchain_community.llms.ai21 import AI21
from langchain_community.llms.vertexai import VertexAI
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ai21.contextual_answers import AI21ContextualAnswers

db_regular = PGVector.from_existing_index(
    embedding=VertexAIEmbeddings(),
    connection_string=os.environ["CONNECTION_STRING_MED_REGULAR"],
)
db_ai21 = PGVector.from_existing_index(
    embedding=VertexAIEmbeddings(),
    connection_string=os.environ["CONNECTION_STRING_MED_AI21"],
)

retriever_ai21 = db_ai21.as_retriever()
retriever_regular = db_regular.as_retriever()

ai21 = AI21(ai21_api_key=os.environ["AI21_API_KEY"])
palm2 = VertexAI(model_name="text-bison@001")

ai21_contextual_answers = AI21ContextualAnswers(
    ai21_api_key=os.environ["AI21_API_KEY"], model="answer"
)


def ask_bot2_ai21_segmented(question_query: str) -> str:
    relevant_docs = db_ai21.similarity_search(query=question_query, k=10)
    texts = " ,".join([doc.page_content for doc in relevant_docs])

    response = ai21_contextual_answers.invoke(question_query, context=texts)
    return response


def ask_bot_simple_textsplitting(question_query: str) -> str:
    prompt = ChatPromptTemplate.from_template(question_query)

    chain = (
        {"context": db_regular.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | palm2
        | StrOutputParser()
    )
    return chain.invoke(question_query)
