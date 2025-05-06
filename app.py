import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in .env file.")
    st.stop()

genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    if not pdf_docs:
        raise ValueError("No PDF files uploaded.")
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            st.warning(f"Error reading PDF {pdf.name}: {str(e)}")
    if not text.strip():
        raise ValueError("No text extracted from PDFs.")
    return text

def get_text_chunks(text):
    """Split text into chunks for vectorization."""
    if not text:
        raise ValueError("Input text is empty.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    if not chunks:
        raise ValueError("No text chunks created.")
    return chunks

def get_vector_store(text_chunks):
    """Create and save FAISS vector store from text chunks."""
    if not text_chunks:
        raise ValueError("Text chunks are empty.")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        # Verify embeddings generation
        embeddings_list = embeddings.embed_documents(text_chunks)
        if not embeddings_list or not embeddings_list[0]:
            raise ValueError("Failed to generate embeddings.")
        
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {str(e)}")

def get_conversational_chain():
    """Create a question-answering chain using Gemini."""
    prompt_template = '''
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say, "Answer is not available in the context." Do not provide incorrect information.
    
    Context: {context}
    Question: {question}
    
    Answer:
    '''
    
    try:
        # Use correct model name (update based on Google's documentation)
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        raise RuntimeError(f"Failed to create conversational chain: {str(e)}")

def user_input(user_question):
    """Process user question and return answer from vector store."""
    if not user_question:
        st.warning("Please enter a question.")
        return
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        if not os.path.exists("faiss_index"):
            st.error("Vector store not found. Please upload and process PDFs first.")
            return
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        if not docs:
            st.write("Reply: No relevant documents found.")
            return
        
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    st.set_page_config(page_title="Chat with PDF (Gemini)", layout="wide")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload PDF Files and Click Submit & Process",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {str(e)}")

if __name__ == "__main__":
    main()