import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

#Page Configuration
st.set_page_config(page_title="Chat with MultiPDF | Gemini AI", layout="wide", initial_sidebar_state="expanded")

#Load API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    st.error("❌ Google API Key is missing! Please check your .env file.")

#Custom Styling for UI
st.markdown("""
    <style>
        body { background: #F5F7FA; }
        .main-title { text-align: center; font-size: 32px; font-weight: bold; color: #4A90E2; }
        .subtitle { text-align: center; font-size: 18px; color: #555; }
        .stTextInput { border: 2px solid #4A90E2; border-radius: 10px; }
        .stButton>button { background: #4A90E2; color: white; border-radius: 8px; width: 100%; font-size: 16px; }
        .stButton>button:hover { background: #357ABD; }
        .sidebar { background: #E3EAF2; padding: 20px; border-radius: 10px; }
        .footer { text-align: center; padding: 20px; font-size: 14px; color: #888; margin-top: 50px; }
    </style>
""", unsafe_allow_html=True)

#Extract text from PDFs
def extract_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""  
    return text

#Split text into smaller chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return splitter.split_text(text)

#Create & save FAISS vector store
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    store = FAISS.from_texts(chunks, embedding=embeddings)
    store.save_local("faiss_index")

#Load FAISS vector store
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#Generate response using AI (Optimized for Speed)
def generate_response(question):
    try:
        store = load_vector_store()
        docs_with_scores = store.similarity_search_with_score(question, k=3)  # Retrieve top 3 relevant chunks
        docs = [doc[0] for doc in docs_with_scores if doc[1] < 0.7]  # Filter low-score results

        if not docs:
            st.warning("No relevant information found.")
            return

        prompt = """
        Answer the question based on the given context.
        If the answer is not available, say: "Answer is not available in the context."
        
        Context:\n {context}\n
        Question:\n {question}\n
        Answer:
        """
        
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1)  # Lowered temp for fast & accurate answers
        qa_chain = load_qa_chain(model, chain_type="stuff", prompt=PromptTemplate(template=prompt, input_variables=["context", "question"]))
        
        response = qa_chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        response_text = response.get("output_text", "No response generated.")
        
        st.success("✅ AI Response Generated")
        st.write("### Reply:", response_text)

        #Add Download Button
        if response_text:
            st.download_button("📥 Download Response", response_text, file_name="AI_Response.txt")

    except Exception as e:
        st.error(f"Error: {e}")

#Streamlit UI
def main():
    #Initialize session state for recent questions
    if "history" not in st.session_state:
        st.session_state.history = []

    #Main Title
    st.markdown('<h1 class="main-title">Chat with MultiPDF | Gemini AI💡</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload PDFs & Ask Questions 💬</p>', unsafe_allow_html=True)

    #User Question Input
    user_question = st.text_input("🔍 Ask a question about the uploaded PDF(s):", placeholder="Type your question here...")
    
    if user_question:
        st.session_state.history.append(user_question)  # ✅ Store question in history
        generate_response(user_question)

    #Show Last 5 Questions
    if st.session_state.history:
        st.markdown("### 🔄 Recent Questions")
        for q in st.session_state.history[-5:]:  
            st.markdown(f"✔ {q}")

    #Sidebar for File Upload
    with st.sidebar:
        st.markdown('<div class="sidebar">', unsafe_allow_html=True)
        st.title("📂 Upload PDFs")
        pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

        if st.button("📌 Process PDFs"):
            if not pdf_files:
                st.warning("⚠️ Please upload at least one PDF.")
                return

            with st.spinner("⏳ Processing your documents..."):
                text = extract_text(pdf_files)
                chunks = split_text(text)
                create_vector_store(chunks)
                st.success("✅ Processing complete! You can now ask questions.")
        
        #Reset Button (Fixed)
        if st.button("🔄 Reset Chat"):
            st.session_state.history = []
            st.rerun()  # ✅ Fixed: Used st.rerun() instead of deprecated st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    #Footer (Centered at the end)
    st.markdown("<div class='footer'>© 2025 Chat with MultiPDF using Gemini by Mayur Satpute - All Rights Reserved 🌐</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()