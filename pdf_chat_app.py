import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
import os

# Load environment variables from .env file
load_dotenv()

# Initialize embeddings without runtime download
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Function to read the uploaded PDF
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the PDF content into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and store FAISS vector store from text chunks
def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

# Function to run the conversation chain with the retrieved data
def get_conversational_chain(tools, ques):
    # Use OpenAI model (make sure the API key is set in environment variables)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    # Set up prompt for the chatbot
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer the question as detailed as possible from the provided context."),
            ("human", "{input}"),
        ]
    )
    
    # Initialize the agent for tool calling
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Process the question and return the response
    response = agent_executor.invoke({"input": ques})
    st.write("Reply:", response['output'])

# Function to handle user input and retrieval
def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool extracts answers from the PDF.")
    get_conversational_chain(retrieval_chain, user_question)

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("RAG-based Chat with PDF")

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF files and click on Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_doc is not None and len(pdf_doc) > 0:
                with st.spinner("Processing..."):
                    raw_text = pdf_read(pdf_doc)
                    text_chunks = get_chunks(raw_text)
                    vector_store(text_chunks)
                    st.success("Done! You can now ask questions.")
            else:
                st.error("Please upload at least one PDF file.")

    # User input section
    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

# Run the app
if __name__ == "__main__":
    main()
