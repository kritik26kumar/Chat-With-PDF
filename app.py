import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        return text
    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")
        return ""

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Cached function to create and save vector store
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY, model="models/embedding-001"
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


# Function to create conversational chain based on selected model
def get_conversational_chain(selected_model):
    prompt_template = '''
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, 
    just say "answer is not available in the context". Do not make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    '''
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    try:
        if selected_model == "Gemini":
            if not GOOGLE_API_KEY:
                st.error("Google API key is missing.")
                return None
            model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        elif selected_model == "Groq":
            if not GROQ_API_KEY:
                st.error("Groq API key is missing.")
                return None
            model = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama3-8b-8192",  # Can be made configurable
                temperature=0.3
            )
        else:
            st.error("Invalid model selected.")
            return None

        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None



# Function to process user input and generate response
def user_input(user_question, selected_model):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY, model="models/embedding-001"
        )
        # Load FAISS index
        if not os.path.exists("faiss_index"):
            st.error("Vector store not found. Please process PDFs first.")
            return

        new_db = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain(selected_model)
        if chain is None:
            return

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.header("Chat with PDFs")

    # Model selection
    selected_model = st.selectbox("Choose a Model:", ["Gemini", "Groq"])
    st.write(f"Using model: {selected_model}")

    # User question input
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question, selected_model)

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        vector_store = get_vector_store(text_chunks)
                        if vector_store:
                            st.success("PDFs processed successfully!")
                    else:
                        st.error("No text chunks generated from PDFs.")
                else:
                    st.error("No text extracted from PDFs.")

if __name__ == "__main__":
    main()
