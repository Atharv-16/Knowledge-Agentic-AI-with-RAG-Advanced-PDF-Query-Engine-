import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Configure Google AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please add GOOGLE_API_KEY to your .env file.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Google AI: {str(e)}")
    st.stop()

# Constants
EMBEDDING_MODEL = "models/embedding-001"
# Updated to a currently available model as of March 2025
CHAT_MODEL = "gemini-1.5-pro"  # Changed from "gemini-pro"
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000
TEMPERATURE = 0.3

def get_pdf_text(pdf_docs):
    """Extract text from PDF documents with error handling."""
    text = ""
    if not pdf_docs:
        return text
        
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"Error reading PDF: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into chunks with improved parameters."""
    if not text.strip():
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create and save vector store with error handling."""
    if not text_chunks:
        raise ValueError("No text chunks to process")
        
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    """Create QA chain with improved prompt template."""
    prompt_template = """
    You are an expert AI assistant trained to answer questions based on provided documents.
    Follow these guidelines:
    1. Answer the question as detailed as possible using the provided context
    2. If the answer isn't in the context, say "I couldn't find this information in the documents"
    3. Always maintain a helpful and professional tone
    4. For complex questions, break down your answer into clear points
    
    Context:\n{context}\n
    Question: {question}\n
    Answer:
    """
    
    try:
        model = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            temperature=TEMPERATURE,
            google_api_key=GOOGLE_API_KEY
        )
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    except Exception as e:
        st.error(f"Failed to initialize chat model: {str(e)}")
        raise

def process_user_input(user_question):
    """Process user question and return response."""
    if not user_question.strip():
        return "Please enter a valid question."
        
    if not os.path.exists("faiss_index"):
        return "No processed documents found. Please upload and process PDFs first."
        
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = vector_store.similarity_search(user_question, k=4)
        
        if not docs:
            return "No relevant information found in documents for this question."
            
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        raise Exception(f"Error processing question: {str(e)}")

def main():
    """Main application function with enhanced UI."""
    st.set_page_config(
        page_title="Knowledge Agentic AI with RAG: Advanced PDF Query Enginet",
        page_icon="💡",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main {padding: 2rem;}
        .stButton>button {width: 100%; background-color: #1f77b4; color: white;}
        .stFileUploader {border: 1px dashed #1f77b4; padding: 1rem;}
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Knowledge Agentic AI with RAG: Advanced PDF Query Engine💡")
    st.markdown("Upload PDFs and ask questions about their content")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about your PDFs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.spinner("Processing..."):
            try:
                response = process_user_input(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception as e:
                error_msg = f"Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
                st.error(error_msg)
    
    # Sidebar
    with st.sidebar:
        st.header("Document Processing")
        
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if st.button("Process Documents"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("No text extracted from PDFs. They might be scanned images.")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Processing failed: {str(e)}")
        
        st.markdown("---")
        st.markdown("""
        ### About
        Chat with multiple PDF documents using Google's Gemini AI.
        Created by [Atharv Gaur](https://github.com/Atharv-16)
        """)

if __name__ == "__main__":
    main()