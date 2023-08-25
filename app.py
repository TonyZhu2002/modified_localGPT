import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os

import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

from run_localGPT import load_model


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
    
def get_conversation_chain(vectorstore):
    device_type = "cuda"
    model_id = "TheBloke/Llama-2-7B-Chat-GGML"
    model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        input_key='question', 
        output_key='answer',
        return_messages=True
    )
    llm = load_model(device_type, model_id=model_id, model_basename=model_basename)

    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']
    st.session_state.citation = response['source_documents']

    for i, message in enumerate(reversed(st.session_state.chat_history), 1):
        if i % 2 != 0:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def handle_citation(user_question):
    for i, cite in enumerate(st.session_state.citation):
        st.markdown(f"##### Source {i+1}")
        st.write(cite.page_content)


def main():
    st.set_page_config(page_title="AI Reading Tool",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "citation" not in st.session_state:
        st.session_state.citation = None

    st.header("Let AI Helps You to Read! :books:")
    user_question = st.text_input("Ask a question about your documents:")

    chat_tab, source_tab = st.tabs(["Chat", "Citation"])

    with chat_tab:
        if user_question:
            handle_userinput(user_question)

    with source_tab:
        if user_question:
            handle_citation(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Injest'", 
                                    accept_multiple_files=True)
        if st.button("Injest"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)





if __name__ == '__main__':
    main()