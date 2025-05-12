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
import streamlit as st

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class business_func:
    # Function to split text into chunks    
    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks

    # Cached function to create and save vector store
    def get_vector_store(self, text_chunks):
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
    def get_conversational_chain(self, selected_model):
        prompt_template = '''
        You are a helpful assistant. Use the provided context extracted from PDF or image files to answer the question as accurately as possible.

        Instructions:
        - Base your answer strictly on the context below.
        - Do NOT guess or fabricate any information.
        - If the answer is not explicitly mentioned in the context, reply: "The answer is not available in the provided context."

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
                    model_name="llama3-8b-8192",
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
    def user_input(self, user_question, selected_model):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY, model="models/embedding-001"
            )
            # Load FAISS index
            if not os.path.exists("faiss_index"):
                st.error("Vector store not found. Please process PDFs first.")
                return None

            new_db = FAISS.load_local(
                "faiss_index", embeddings, allow_dangerous_deserialization=True
            )
            docs = new_db.similarity_search(user_question)

            chain = self.get_conversational_chain(selected_model)
            if chain is None:
                return None

            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            response_text = response["output_text"]
            return response_text
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            return None