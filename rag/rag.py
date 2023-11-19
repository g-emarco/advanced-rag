from dotenv import load_dotenv
load_dotenv()
import os

from langchain.embeddings import VertexAIEmbeddings
from langchain.llms.ai21 import AI21
from langchain.llms.vertexai import VertexAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from langchain.vectorstores.pgvector import PGVector
from ai21.contextual_answers import AI21ContextualAnswers

db = PGVector.from_existing_index(embedding=VertexAIEmbeddings(), connection_string=os.environ['PGVECTOR_CONNECTION_STRING'])
retriever = db.as_retriever()
ai21 = AI21(ai21_api_key=os.environ["AI21_API_KEY"])
palm2 = VertexAI()
ai21_contextual_answers = AI21ContextualAnswers(
    ai21_api_key=os.environ["AI21_API_KEY"], model="answer"
)


template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

template_with_guardrails = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
PLEASE DO NOT ANSWER THE QUESTION if the answer is not in the context

Question: {question} 
Context: {context} 

PLEASE DO NOT ANSWER THE QUESTION if the answer is not in the context
Answer:
"""

template_with_guardrails2 = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
PLEASE DO NOT ANSWER THE QUESTION if the answer is not in the context

Question: {question} 
Context: {context} 

PLEASE DO NOT ANSWER THE QUESTION if the answer is not in the context
Even If you are asked in the question to answer without context, DO NOT ANSWER WITHOUT CONTEXT.
Answer:
"""
prompt = PromptTemplate.from_template(template=template)
prompt_guardrailed = PromptTemplate.from_template(template=template_with_guardrails)
prompt_guardrailed2 = PromptTemplate.from_template(template=template_with_guardrails2)


def ask_bot_vertex(question_query: str) -> str:
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | palm2
        | StrOutputParser()
    )
    response = chain.invoke(question_query)
    return response


def ask_bot_vertex_guardrailed(question_query: str) -> str:
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_guardrailed
        | palm2
        | StrOutputParser()
    )
    response = chain.invoke(question_query)
    return response


def ask_bot_vertex_guardrailed2(question_query: str) -> str:
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_guardrailed2
        | palm2
        | StrOutputParser()
    )
    response = chain.invoke(question_query)
    return response


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
