import streamlit as st
from Business_Access_Layer.Bal import business_func as bf
from Data_Access_Layer.Dal import data_loader as dl

obj_bal=bf()
obj_dal=dl()

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
        obj_bal.user_input(user_question, selected_model)

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
                raw_text = obj_dal.get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = obj_bal.get_text_chunks(raw_text)
                    if text_chunks:
                        vector_store = obj_bal.get_vector_store(text_chunks)
                        if vector_store:
                            st.success("PDFs processed successfully!")
                    else:
                        st.error("No text chunks generated from PDFs.")
                else:
                    st.error("No text extracted from PDFs.")

if __name__ == "__main__":
    main()