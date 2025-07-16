import os
import streamlit as st
from dotenv import load_dotenv
from typing import Any, Dict, List
from pydantic import SecretStr
import asyncio
import functools
import threading

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from utils.timeline_extractor import extract_timeline
import plotly.express as px  # type: ignore
import pandas as pd  # type: ignore
import json

from utils.pdf_parser import extract_text_from_pdf

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ Google API Key not found. Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

# Function to reset the input field
def clear_input():
    # Increment key to force input widget to reset
    st.session_state.input_key += 1

def create_embeddings_and_vectorstore(_text_chunks, _api_key):
    """Create embeddings and vector store in a way that handles async issues"""
    import threading
    import asyncio
    
    def run_in_thread():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize embeddings with proper configuration
            embedding = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=_api_key
            )
            
            # Create vector store
            vectorstore = FAISS.from_documents(_text_chunks, embedding)
            vectorstore.save_local("faiss_index")
            
            return vectorstore
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return None
        finally:
            loop.close()
    
    # Run the embedding creation in a separate thread to avoid event loop conflicts
    result = [None]
    exception = [None]
    
    def thread_target():
        try:
            result[0] = run_in_thread()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join()
    
    if exception[0]:
        st.error(f"Error creating embeddings: {str(exception[0])}")
        return None
    
    return result[0]

def create_qa_chain(vectorstore):
    """Create QA chain with Gemini LLM"""
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", 
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff"
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Historical Timeline Chatbot", layout="wide")

# Custom CSS for better chat styling
st.markdown("""
<style>
    /* Main app styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Chat container styling */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1.5rem;
        border: 2px solid #f0f2f6;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        background: #fafafa;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Message styling */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 1rem 0;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 25px 25px 5px 25px;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0 3px 15px rgba(102, 126, 234, 0.3);
        font-size: 0.95rem;
        line-height: 1.4;
    }
    
    .bot-message {
        display: flex;
        justify-content: flex-start;
        margin: 1rem 0;
        align-items: flex-start;
    }
    
    .bot-avatar {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        flex-shrink: 0;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .bot-bubble {
        background: white;
        color: #333;
        padding: 15px 20px;
        border-radius: 25px 25px 25px 5px;
        max-width: 75%;
        word-wrap: break-word;
        border: 2px solid #f0f2f6;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Input area styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e6e9ef;
        padding: 12px 20px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 3px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 15px;
        border: 2px dashed #667eea;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
    
    /* Status messages */
    .stSuccess {
        border-radius: 15px;
        background: linear-gradient(135deg, #00c851 0%, #007e33 100%);
        color: white;
    }
    
    .stInfo {
        border-radius: 15px;
        background: linear-gradient(135deg, #33b5e5 0%, #0099cc 100%);
        color: white;
    }
    
    .stWarning {
        border-radius: 15px;
        background: linear-gradient(135deg, #ffbb33 0%, #ff8800 100%);
        color: white;
    }
    
    /* Welcome message */
    .welcome-message {
        text-align: center;
        padding: 3rem 2rem;
        color: #666;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Processing status */
    .processing-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="app-header">
    <div class="app-title">📚 Historical Timeline Chatbot</div>
    <div class="app-subtitle">Upload any historical PDF and chat with an AI to explore timelines and events</div>
</div>
""", unsafe_allow_html=True)

# Upload PDF
with st.container():
    st.markdown("### 📎 Upload Your Historical Document")
    pdf_file = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"],
        help="Upload any historical textbook, research paper, or archive document"
    )

if pdf_file is not None:
    # Check if this is a new file
    if st.session_state.processed_file != pdf_file.name:
        with st.spinner("📖 Processing PDF... This may take a moment."):
            try:
                # Create data directory if it doesn't exist
                os.makedirs("data", exist_ok=True)
                
                # Save uploaded PDF
                file_path = f"data/{pdf_file.name}"
                with open(file_path, "wb") as f:
                    f.write(pdf_file.getvalue())

                # Extract text
                raw_text = extract_text_from_pdf(file_path)
                
                if not raw_text or len(raw_text.strip()) < 100:
                    st.error("❌ Could not extract sufficient text from the PDF. Please check if the PDF contains readable text.")
                    st.stop()

                # Timeline Extraction
                try:
                    timeline_data = extract_timeline(raw_text)
                    
                    # Optional: Save to JSON
                    json_path = f"data/{pdf_file.name}_timeline.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(timeline_data, f, indent=2)

                    # Show download button
                    if timeline_data:
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            st.download_button(
                                "📥 Download Timeline JSON", 
                                file_name="timeline.json", 
                                mime="application/json",
                                data=json.dumps(timeline_data, indent=2),
                                use_container_width=True
                            )
                except Exception as e:
                    st.warning(f"Timeline extraction failed: {str(e)}. Continuing with RAG setup...")

                # Split text into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                )
                chunks = splitter.create_documents([raw_text])
                
                if not chunks:
                    st.error("❌ Could not create text chunks from the PDF.")
                    st.stop()

                # Print chunk information to terminal
                print(f"📄 Created {len(chunks)} text chunks for processing (chunk_size=1000, overlap=100)")
                print(f"📊 Total characters in PDF: {len(raw_text):,}")
                print(f"📑 Average chunk size: {sum(len(chunk.page_content) for chunk in chunks) // len(chunks)} characters")

                # Check if FAISS index already exists to avoid reprocessing
                if os.path.exists("faiss_index") and st.session_state.processed_file == pdf_file.name:
                    st.info("✅ Using existing FAISS index.")
                    # Load existing vector store
                    embedding = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=GOOGLE_API_KEY
                    )
                    vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
                else:
                    # Create embeddings and vector store
                    vectorstore = create_embeddings_and_vectorstore(chunks, GOOGLE_API_KEY)
                
                if vectorstore is None:
                    st.error("❌ Failed to create vector store.")
                    st.stop()

                # Create QA chain
                qa_chain = create_qa_chain(vectorstore)
                
                if qa_chain is None:
                    st.error("❌ Failed to create QA chain.")
                    st.stop()

                # Store in session state
                st.session_state.qa_chain = qa_chain
                st.session_state.processed_file = pdf_file.name

                st.success("✅ PDF processed successfully! You can now ask questions about the content.")
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.stop()
    else:
        st.info("✅ PDF already processed. You can ask questions below.")

    # Chat Interface
    if st.session_state.qa_chain is not None:
        st.markdown("### 💬 Chat with Your Document")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for i, (user_msg, bot_msg, sources) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(
                    f'<div class="user-message">'
                    f'<div class="user-bubble">'
                    f'<strong>You</strong><br>{user_msg}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
                
                # Bot message with avatar
                st.markdown(
                    f'<div class="bot-message">'
                    f'<div class="bot-avatar">🤖</div>'
                    f'<div class="bot-bubble">'
                    f'<strong>AI Assistant</strong><br>{bot_msg}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="welcome-message">
                <div class="welcome-icon">🎯</div>
                <strong>Ready to explore your document!</strong><br>
                Ask me anything about the historical content you've uploaded.<br>
                Try questions like: "What happened in 1947?" or "Tell me about the main events"
            </div>
            """, unsafe_allow_html=True)
        
        # Input area
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "💭 Ask me anything about your document...", 
                placeholder="e.g., What were the major events in 1947? Who were the key figures?",
                key=f"chat_input_{st.session_state.input_key}",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send �", use_container_width=True, key="send_btn")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_btn", use_container_width=True):
                st.session_state.chat_history = []
                clear_input()
                st.rerun()
        
        # Handle message sending - only when send button is clicked
        if send_button and user_input and user_input.strip():
            question_to_send = user_input  # Store the current input
            
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.qa_chain({"query": question_to_send})
                    
                    # Add to chat history (sources removed for cleaner UI)
                    st.session_state.chat_history.append((
                        question_to_send,
                        response["result"],
                        []  # Empty sources list
                    ))
                    
                    # Clear the input by changing its key
                    clear_input()
                    
                    # Rerun to update the UI with the new chat history and cleared input
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
        
        # Auto-scroll to bottom (removed problematic JavaScript)
        if st.session_state.chat_history:
            st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

else:
    # Landing page when no PDF is uploaded
    st.markdown("""
    <div class="welcome-message">
        <div class="welcome-icon">📚</div>
        <h3>Welcome to Historical Timeline Chatbot!</h3>
        <p>Upload any historical PDF document to get started.</p>
        <p><strong>What you can do:</strong></p>
        <p>📄 Upload historical textbooks, research papers, or archive documents<br>
        🤖 Ask questions about dates, events, people, and places<br>
        💾 Download extracted timeline data as JSON<br>
        🎯 Get AI-powered answers from your documents</p>
    </div>
    """, unsafe_allow_html=True)
