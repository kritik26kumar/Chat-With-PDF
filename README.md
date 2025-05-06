📚 Ask to ChatGPT – Multi-PDF RAG Chatbot

A Streamlit-based chatbot that allows you to upload multiple PDF files and ask questions about their contents. It uses LangChain, FAISS, and Gemini (Google Generative AI) for Retrieval-Augmented Generation (RAG).

---

🚀 Features

- 📄 Upload multiple PDFs
- ❓ Ask natural language questions
- 🤖 Answers are generated using context-aware retrieval from the PDFs
- 💬 Powered by LangChain + Google Generative AI (Gemini)
- 🧠 Embeddings stored and retrieved using FAISS vector store
- 🖥️ Deployed with Streamlit

---

📦 Tech Stack

- Streamlit
- LangChain
- Google Generative AI (Gemini)
- FAISS
- PyPDF2
- dotenv

---

📁 Project Structure

.
├── app.py                  # Streamlit app
├── requirements.txt        # Required Python packages
├── .env                    # API Key (not pushed to GitHub)
├── faiss_index/            # Vector store (auto-generated)
└── README.md               # Project documentation

---

🔧 Setup Instructions

1. Clone the Repository

    git clone https://github.com/kritik26kumar/Ask-to-ChatGPT-Chatbot-.git
    cd Ask-to-ChatGPT-Chatbot-

2. Create Virtual Environment (Optional but Recommended)

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate

3. Install Dependencies

    pip install -r requirements.txt

4. Add Your API Key

Create a `.env` file in the root folder:

    GOOGLE_API_KEY=your_google_generative_ai_api_key_here

> 🔒 Never commit your API key to GitHub!

---

▶️ Run the App Locally

    streamlit run app.py

Open the link shown in your terminal (usually http://localhost:8501)

---

☁️ Deployment (Streamlit Cloud)

1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New App", select your repo, and set `app.py` as the entry file
4. Add your GOOGLE_API_KEY in Secrets (in the settings tab)

---

🧠 Example Use Cases

- Academic research PDF Q&A
- Legal document search
- Business reports summarization
- Technical documentation assistant

---

📬 Contact

Feel free to reach out via GitHub Issues for feedback or questions!

---

⭐ Star this repo if you found it helpful!
"""
