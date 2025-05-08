from PyPDF2 import PdfReader
import streamlit as st

class data_loader():
    # Function to extract text from PDFs
    def get_pdf_text(self,pdf_docs):
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