import streamlit as st
from PyPDF2 import PdfReader  # Or use PyMuPDF for faster performance
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None

def get_embeddings():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return OpenAIEmbeddings(openai_api_key=openai_api_key)

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    embeddings = get_embeddings()
    st.session_state['vector_store'] = FAISS.from_texts(text_chunks, embedding=embeddings)

def get_conversational_chain(retriever, ques):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    tools = [Tool(name="pdf_extractor", func=retriever.get_relevant_documents, description="Extracts information from the PDF.")]
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    response = agent.run(ques)
    st.write("Reply:", response)

def user_input(user_question):
    if st.session_state['vector_store'] is None:
        st.error("Please upload and process a PDF first.")
        return
    retriever = st.session_state['vector_store'].as_retriever()
    get_conversational_chain(retriever, user_question)

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("RAG-based Chat with PDF")

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF files and click on Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_doc and len(pdf_doc) > 0:
                with st.spinner("Processing..."):
                    start_time = time.time()
                    raw_text = pdf_read(pdf_doc)
                    text_chunks = get_chunks(raw_text)
                    vector_store(text_chunks)
                    end_time = time.time()
                    st.success(f"Done! Processing took {end_time - start_time:.2f} seconds.")
            else:
                st.error("Please upload at least one PDF file.")

    user_question = st.text_input("Ask a question from the PDF files")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
