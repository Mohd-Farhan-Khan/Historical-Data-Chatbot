# 📚 Historical Timeline Chatbot

A Streamlit application that extracts timelines from historical PDFs and creates an interactive RAG-based chatbot using Google's Gemini 2.0 Flash AI model.

## ✨ Features

- 📄 **PDF Upload**: Upload any history textbook or archive PDF
- � **RAG Implementation**: Retrieval-Augmented Generation for accurate answers
- 🤖 **AI Chatbot**: Natural conversation interface powered by Gemini 2.0 Flash
- 💬 **Chat History**: Persistent chat history with source references
- 💾 **Export**: Download extracted timeline data as JSON

## 🛠️ Setup

### Prerequisites
- Python 3.12 or higher
- Google API Key for Gemini AI (with access to Gemini 2.0 Flash)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/yourusername/Historical-timeline-chatbot.git
   cd Historical-timeline-chatbot
   ```

2. **Set up virtual environment** (Two options)
   
   **Option A**: Use the existing environment
   ```bash
   .\historical_rag_bot\Scripts\activate  # Windows
   ```
   
   **Option B**: Create a fresh environment
   ```bash
   python -m venv historical_rag_bot
   .\historical_rag_bot\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Create a `.env` file in the project root
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```

### 🚀 Running the Application

#### Option 1: Using the VS Code Task (Recommended)
```
Run the "Run Streamlit App" task in VS Code
```

#### Option 2: Direct command
```powershell
.\historical_rag_bot\Scripts\python.exe -m streamlit run app.py
```

#### Option 3: After activating virtual environment
```bash
.\historical_rag_bot\Scripts\activate
python -m streamlit run app.py
```

## 📖 Usage

1. **Start the application** using one of the methods above
2. **Open your browser** to `http://localhost:8501` (opens automatically)
3. **Upload a PDF** containing historical content
4. **Wait for processing** while the app:
   - Extracts text from the PDF
   - Creates timeline data
   - Chunks the text for embedding
   - Creates vector embeddings using Google's embedding model
   - Sets up the RAG system
5. **Chat with your PDF** by asking questions in the chat interface:
   - Questions about dates (e.g., "What happened in 1947?")
   - Questions about events (e.g., "Tell me about the Vedic Period")
   - Questions about figures, places, or concepts in the PDF
6. **View source context** by expanding the sources section below each answer
7. **Download** the extracted timeline as JSON if needed

## 🗂️ Project Structure

```
Historical-timeline-chatbot/
├── app.py                 # Main Streamlit application with RAG implementation
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this - not in git)
├── data/                  # Directory for uploaded PDFs and extracted timeline JSON
├── faiss_index/           # Vector database storage (not in git)
├── utils/
│   ├── pdf_parser.py      # PDF text extraction utility
│   └── timeline_extractor.py  # Timeline extraction logic
└── historical_rag_bot/    # Virtual environment (not in git)
```

## 🔧 Troubleshooting

### Common Issues

1. **Event Loop Error in Thread 'ScriptRunner.scriptThread'**
   - **Solution**: This is handled by the threading implementation in the code

2. **"Google API Key not found"**
   - **Solution**: Ensure your `.env` file contains `GOOGLE_API_KEY=your_key`
   - **Double-check**: Make sure your API key has access to Gemini models

3. **"Failed to create vector store"**
   - **Solution**: Check your internet connection and API key permissions
   - **Alternative**: Try reducing chunk size in the code (currently 1000)

4. **Slow PDF processing**
   - **Solution**: For large PDFs, be patient during initial processing
   - **Note**: Once processed, the same PDF will load quickly on subsequent runs

## 🔒 Security and Privacy

- Your Google API key is stored locally in the `.env` file (not committed to git)
- Uploaded PDFs are stored locally in the `data/` directory
- Vector embeddings are stored locally in the `faiss_index/` directory
- No data is sent to external servers except for Google API calls for embeddings and LLM responses

## 📝 License

MIT License - See LICENSE file for details
   - **Solution**: Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

## Dependencies

- `streamlit` - Web application framework
- `langchain` - LLM framework
- `langchain-google-genai` - Google Gemini integration
- `faiss-cpu` - Vector similarity search
- `PyMuPDF` - PDF text extraction
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation
- `python-dotenv` - Environment variable management

## API Keys

Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## License

This project is for educational and research purposes.
