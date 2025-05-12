import streamlit as st
from Business_Access_Layer.Bal import business_func as bf
from Data_Access_Layer.Dal import data_loader as dl

obj_bal = bf()
obj_dal = dl()

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDFs and Images", layout="wide")
    st.header("📄 Chat with PDFs and Images")

    # Model selection
    selected_model = st.selectbox("Choose a Model:", ["Gemini", "Groq"])
    st.write(f"Using model: **{selected_model}**")

    # Initialize chat history and debug log in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "debug_log" not in st.session_state:
        st.session_state["debug_log"] = []

    # Display chat history (only user and assistant messages)
    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input for user questions
    user_question = st.chat_input("Ask a question about the uploaded files:")
    if user_question:
        # Append user question to chat history
        st.session_state["messages"].append({"role": "user", "content": user_question})
        
        # Display user question immediately
        with st.chat_message("user"):
            st.markdown(user_question)

        # Build context from previous messages (limit to last 5 for brevity)
        context = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" 
             for msg in st.session_state["messages"] 
             if msg["role"] in ["user", "assistant"][-5:-1]]  # Exclude current question
        )
        # Combine context and current question
        full_question = f"Context:\n{context}\n\nCurrent Question:\n{user_question}" if context else user_question

        # Debug: Log question to debug_log and terminal
        st.session_state["debug_log"].append(f"Debug: Sending question: {full_question}")
        print(f"Debug: Sending question: {full_question}")  # Log to terminal

        # Process question with business_func
        try:
            response = obj_bal.user_input(full_question, selected_model)
            # Handle different response cases
            if response is not None and response.strip():
                # Valid response: store and display
                st.session_state["messages"].append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            elif response == "":
                # Empty response: store and display warning
                error_msg = "Empty response received. The model may not have generated an answer."
                st.session_state["messages"].append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.warning(error_msg)
            else:
                # None response: store and display error
                error_msg = "No response generated. Please check if files are processed or try a different question."
                st.session_state["messages"].append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
        except Exception as e:
            # Exception: store and display error
            error_msg = f"Error processing question: {str(e)}"
            st.session_state["messages"].append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)

    # Sidebar for file upload
    with st.sidebar:
        st.title("📂 Menu")
        MAX_FILES = 10  # Set maximum number of files allowed
        uploaded_files = st.file_uploader(
            "Upload your PDF or image files",
            accept_multiple_files=True,
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"]
        )

        if st.button("Submit & Process"):
            if not uploaded_files:
                st.error("Please upload at least one PDF or image file.")
                return

            # Check if the number of files exceeds the limit
            if len(uploaded_files) > MAX_FILES:
                st.error(f"Too many files uploaded. Maximum allowed is {MAX_FILES}. Please upload fewer files.")
                return

            # Only process if not already done
            if "processed" not in st.session_state or not st.session_state["processed"]:
                with st.spinner("Processing files..."):
                    raw_text = ""
                    pdf_files = [f for f in uploaded_files if f.name.lower().endswith('.pdf')]
                    image_files = [f for f in uploaded_files if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

                    # Process PDFs
                    if pdf_files:
                        pdf_text = obj_dal.extract_text_from_pdf(pdf_files)
                        raw_text += pdf_text + "\n" if pdf_text else ""

                    # Process images
                    for image_file in image_files:
                        image_text = obj_dal.extract_text_from_image(image_file)
                        raw_text += image_text + "\n" if image_text else ""

                    if raw_text.strip():
                        text_chunks = obj_bal.get_text_chunks(raw_text)
                        if text_chunks:
                            vector_store = obj_bal.get_vector_store(text_chunks)
                            if vector_store:
                                st.success("✅ Files processed successfully!")
                                st.session_state["processed"] = True
                                st.write(f"📚 Total chunks created: {len(text_chunks)}")
                            else:
                                st.error("Failed to create vector store.")
                        else:
                            st.error("No text chunks generated from the files.")
                    else:
                        st.error("No text extracted from the uploaded files.")
            else:
                st.info("📌 Files already processed. You can ask questions now.")

        # Reset button to clear files, chat history, and debug log
        if st.button("Reset"):
            st.session_state["processed"] = False
            st.session_state["messages"] = []  # Clear chat history
            st.session_state["debug_log"] = []  # Clear debug log
            st.success("🌀 Processing state, chat history, and debug log reset. You can re-upload files.")

if __name__ == "__main__":
    main()