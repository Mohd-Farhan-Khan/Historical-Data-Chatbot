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
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 10px 0;
    }
    
    .user-bubble {
        background-color: #007ACC;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        display: flex;
        justify-content: flex-start;
        margin: 10px 0;
    }
    
    .bot-bubble {
        background-color: #f8f9fa;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        word-wrap: break-word;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    
    .stButton > button {
        border-radius: 20px;
        background-color: #007ACC;
        color: white;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #005a9e;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 Historical Timeline Chatbot")
st.markdown("Upload a history-related PDF and ask any date-based questions (e.g., *What happened in 1947?*).")

# Upload PDF
pdf_file = st.file_uploader("Upload a history textbook or archive PDF", type=["pdf"])

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
                        st.download_button("📥 Download Timeline JSON", file_name="timeline.json", mime="application/json",
                                           data=json.dumps(timeline_data, indent=2))
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
        st.markdown("### 💬 Chat with Your PDF")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for i, (user_msg, bot_msg, sources) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(
                    f'<div class="user-message">'
                    f'<div class="user-bubble">'
                    f'<strong>You:</strong><br>{user_msg}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
                
                # Bot message
                st.markdown(
                    f'<div class="bot-message">'
                    f'<div class="bot-bubble">'
                    f'<strong>🤖 Assistant:</strong><br>{bot_msg}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
                
                # Sources expandable section
                if sources:
                    with st.expander(f"📚 Sources for message {i+1}", expanded=False):
                        for j, source in enumerate(sources):
                            st.markdown(f"**Source {j+1}:**")
                            st.text(source[:500] + "..." if len(source) > 500 else source)
                            if j < len(sources) - 1:
                                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👋 Welcome! Upload a PDF and start chatting to get answers about its content.")
        
        # Input area at the bottom
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your message...", 
                placeholder="Ask about historical events, dates, or any content from your PDF...",
                key=f"chat_input_{st.session_state.input_key}",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send 📤", use_container_width=True, key="send_btn")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", key="clear_btn"):
            st.session_state.chat_history = []
            clear_input()
            st.rerun()
        
        # Handle message sending - only when send button is clicked
        if send_button and user_input and user_input.strip():
            question_to_send = user_input  # Store the current input
            
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.qa_chain({"query": question_to_send})
                    
                    # Extract source documents
                    sources = []
                    if "source_documents" in response and response["source_documents"]:
                        sources = [doc.page_content for doc in response["source_documents"]]
                    
                    # Add to chat history
                    st.session_state.chat_history.append((
                        question_to_send,
                        response["result"],
                        sources
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
    st.info("👆 Please upload a PDF file to get started.")
