from langchain.embeddings import VertexAIEmbeddings
from langchain.llms.vertexai import VertexAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.faiss import FAISS

retriever = FAISS.load_local("faiss_index", VertexAIEmbeddings()).as_retriever()
palm2 = VertexAI()

from langchain import hub

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
