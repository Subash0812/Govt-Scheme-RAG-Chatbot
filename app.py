"""SchemeSathi — a portfolio-ready RAG chatbot for Indian government schemes."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import SchemeRetriever, answer_question

DATA_FILE = Path(__file__).parent / "data" / "schemes.json"
load_dotenv(Path(__file__).parent / ".env")

st.set_page_config(page_title="SchemeSathi | Government Schemes", page_icon="🇮🇳", layout="wide")


@st.cache_resource(show_spinner="Preparing the scheme knowledge base…")
def get_retriever() -> SchemeRetriever:
    return SchemeRetriever.from_json(DATA_FILE)


def reset_chat() -> None:
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🇮🇳 SchemeSathi")
    st.caption("A RAG assistant for Indian government schemes")
    st.divider()
    st.subheader("How it works")
    st.markdown("1. Searches a curated scheme knowledge base\n2. Retrieves the most relevant sources\n3. Answers only from those sources")
    st.divider()
    st.subheader("Try asking")
    for prompt in [
        "How do I get support for LPG connection?",
        "Which scheme helps farmers with income support?",
        "Tell me about health insurance for low-income families.",
        "What housing support is available in rural areas?",
    ]:
        if st.button(prompt, use_container_width=True):
            st.session_state.pending_question = prompt
    st.button("Clear conversation", on_click=reset_chat, use_container_width=True)
    st.divider()
    st.caption("Knowledge base: curated sample of Central schemes. Verify details and apply through official portals.")

st.title("Find the right government scheme, in plain English")
st.caption("Ask about benefits, broad eligibility, documents, and how to apply.")
st.info("This is an educational portfolio project—not an official government service or final eligibility decision.", icon="ℹ️")

retriever = get_retriever()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.markdown(f"- [{source['title']}]({source['url']})")

question = st.chat_input("Example: What schemes can help a woman start a small business?")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching scheme information…"):
            result = answer_question(question, retriever)
        st.markdown(result.answer)
        with st.expander("Sources used", expanded=True):
            for source in result.sources:
                st.markdown(f"- [{source['title']}]({source['url']})")
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )
