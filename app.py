from dotenv import load_dotenv

from rag.rag import (
    ask_bot_vertex,
    ask_bot_vertex_guardrailed,
    ask_bot_vertex_guardrailed2,
    ask_bot_ai21,
    ask_bot2_ai21,
)
import os

if not os.environ["PRODUCTION"]:
    load_dotenv()

import streamlit as st

print("***Lemonade 🍋 Documentation Helper***")


st.set_page_config(
    page_title="Chat with Lemonade docs",
    page_icon="🍋",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Lemonade 🍋 Documentation Helper")

AI21 = "AI21 Contextual Answers"
PALM2 = "PaLM 2"
PALM2_GUARDRAILED = "PaLM 2 with protective prompt"
PALM2_GUARDRAILED2 = "PaLM 2 with LLM Guard"

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Lemonade's FAQ Page?\n"
            "https://www.lemonade.com/faq",
        }
    ]

model = st.radio("model", [PALM2, PALM2_GUARDRAILED, PALM2_GUARDRAILED2, AI21], index=0)

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
