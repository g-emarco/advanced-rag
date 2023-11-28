from dotenv import load_dotenv

from rag.rag import (
    ask_bot_vertex,
    ask_bot_vertex_guardrailed,
    ask_bot_vertex_guardrailed2,
    ask_bot_ai21,
    ask_bot2_ai21,
    template,
    template_with_guardrails,
)
import os
from PIL import Image

if not os.environ.get("PRODUCTION"):
    load_dotenv()

import streamlit as st

print("***Lemonade 🍋 Documentation Helper***")


st.set_page_config(
    page_title="Chat with Lemonade docs",
    page_icon="🍋",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

st.title("Lemonade 🍋 Documentation Helper")

AI21 = "AI21 Contextual Answers"
PALM2 = "PaLM 2"
PALM2_GUARDRAILED = "PaLM 2 with protective prompt"
PALM2_GUARDRAILED2 = "PaLM 2 with LLM Guard"


with st.sidebar:
    model = st.radio(
        "model", [PALM2, PALM2_GUARDRAILED, PALM2_GUARDRAILED2, AI21], index=0
    )
    if model == PALM2:
        prompt = template
        st.code(prompt, language="python")

    if model == PALM2_GUARDRAILED:
        prompt = template_with_guardrails
        st.code(prompt, language="python")

        code = """
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | palm2
                | StrOutputParser()
            )
            response = chain.invoke(question_query)
            """

    if model == PALM2_GUARDRAILED2:
        code = """chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_guardrailed
        | palm2
        | StrOutputParser()
    )
    response = chain.invoke(question_query)

    from llm_guard.output_scanners import Toxicity

    scanner = Toxicity(threshold=0.2)
    sanitized_output, is_valid, risk_score = scanner.scan(
        prompt_guardrailed.template, response
    )

    if not is_valid:
        return "LLM Guard found the response as toxic"

    return sanitized_output
        """
        st.code(code, language="python")

    if model == AI21:
        prompt = template
        st.code(prompt, language="python")

        code = """
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | AI21ContextualAnswers(ai21_api_key=os.environ["AI21_API_KEY"], model="answer")
                | StrOutputParser()
            )
            response = chain.invoke(question_query)
        """
        st.code(code, language="python")

tab1, tab2, tab3, tab4 = st.tabs([PALM2, PALM2_GUARDRAILED, PALM2_GUARDRAILED2, AI21])
with tab1:
    left_co, cent_co, last_co = st.columns(3)
    with left_co:
        image = Image.open("static/palm231.webp")
        st.image(image)
    model = PALM2

with tab2:
    left_co, cent_co, last_co = st.columns(3)
    with left_co:
        image = Image.open("static/palm231.webp")
        st.image(image)
    model = PALM2_GUARDRAILED

with tab3:
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        image = Image.open("static/llmguardlogo.png")
        st.image(image)
    with left_co:
        image = Image.open("static/palm231.webp")

        st.image(image)
    model = PALM2_GUARDRAILED2

with tab4:
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        image = Image.open("static/ai221.png")
        st.image(image, caption="Contextual Answers")
    model = AI21

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Lemonade's FAQ Page?\n"
            "https://www.lemonade.com/faq",
        }
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if model == AI21:
                response = ask_bot_ai21(question_query=prompt)
            if model == PALM2:
                response = ask_bot_vertex(question_query=prompt)
            if model == PALM2_GUARDRAILED:
                response = ask_bot_vertex_guardrailed(question_query=prompt)
            if model == PALM2_GUARDRAILED2:
                response = ask_bot_vertex_guardrailed2(question_query=prompt)

            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
