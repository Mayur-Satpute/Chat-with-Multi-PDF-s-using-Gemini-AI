# 💬 Chat with MultiPDF using Gemini AI

A Streamlit web application that lets you **upload multiple PDF documents** and **ask questions** using **Google Gemini AI**. The app uses **LangChain**, **FAISS**, and **Google Generative AI embeddings** to give smart, accurate answers based on your documents.

## 🚀 Features

- 📄 Upload and process multiple PDF files.
- 🔍 Ask questions related to the uploaded content.
- 🧠 Powered by Gemini 1.5 Pro and Google Embeddings.
- 🧾 Responses are generated from document context using vector search.
- 💾 Download AI-generated answers.
- 🔁 View recent questions and reset the chat history.
- 🎨 Clean and responsive Streamlit UI.

## 📸 Demo
![Screenshot (14)](https://github.com/user-attachments/assets/00dbba16-016f-4dca-84c7-cd52b0d2ff32)



## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: LangChain + Google Generative AI
- **Vector Store**: FAISS
- **PDF Parsing**: PyPDF2
- **Environment Management**: dotenv

## 🧑‍💻 How It Works

1. Upload one or more PDF files.
2. Text is extracted and split into chunks.
3. Chunks are embedded using Gemini AI embeddings.
4. Stored in a local FAISS vector store.
5. Your question is matched with relevant chunks.
6. Gemini 1.5 Pro answers based on that context.

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/chat-with-multipdf-gemini.git
cd chat-with-multipdf-gemini

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔐 Setup

1. Create a `.env` file in the root directory:
   ```
   GOOGLE_API_KEY="your-google-api-key"
   ```
2. Replace `"your-google-api-key"` with your actual [Google Generative AI API Key](https://aistudio.google.com/app/apikey).

## 🧪 Run the App

```bash
streamlit run app.py
```

## 📁 Folder Structure

```
.
├── app.py                # Main Streamlit app
├── requirements.txt      # Required Python packages
├── .env                  # Contains API key (not to be shared)
└── faiss_index/          # Auto-created: stores vector embeddings
```

## ⚠️ Notes

- Ensure you have internet access to call Gemini APIs.
- Make sure `GOOGLE_API_KEY` is valid and usage limits are respected.
- Don’t expose your `.env` file in public repositories.

## 🧑‍🎓 Author

**Mayur Satpute**  
Aspiring Frontend & AI Developer  
[LinkedIn](https://www.linkedin.com/in/mayur7pute) | [GitHub](https://github.com/Mayur-Satpute)

