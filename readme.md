# Enhanced RAG Chatbot with Database Integration

A powerful on-premise chatbot that combines Retrieval-Augmented Generation (RAG) with database querying capabilities. Built with Streamlit, LangChain, and Ollama, this solution provides a secure, locally-hosted alternative to cloud-based chatbots.

## Features

- 🔒 **Fully On-Premise**: All components run locally for maximum data security
- 📄 **Document Processing**: Support for PDF, TXT, CSV, XLSX, DOC(X), PPT(X)
- 💾 **Database Integration**: Connect to PostgreSQL, MySQL, MS SQL Server, or Oracle.
- 🔍 **Vector Search**: Efficient document retrieval using LanceDB
- 🤖 **Multiple LLM Support**: Compatible with any model available through Ollama
- 🔄 **Natural Language to SQL**: Convert natural language queries to SQL- Future scope
- 📊 **Interactive UI**: Built with Streamlit for easy interaction

## Prerequisites

- Python 3.8 or higher
- Ollama installed and running locally
- Required database drivers (based on your needs):
  - PostgreSQL: `psycopg2`
  - MySQL: `pymysql`
  - MS SQL Server: `pyodbc` + ODBC Driver 17
  - Oracle: `cx_Oracle`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag_chatbot.git
cd enhanced_rag_chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install Ollama from [https://ollama.ai](https://ollama.ai)

5. Pull your preferred language model (e.g., Llama 2):
```bash
ollama pull llama2
```

## Configuration

1. Create required directories:
```bash
mkdir persistent_storage temp_storage
```

2. (Optional) Configure database drivers:

   - **PostgreSQL**:
     ```bash
     pip install psycopg2-binary
     ```
   
   - **MySQL**:
     ```bash
     pip install pymysql
     ```
   
   - **MS SQL Server**:
     ```bash
     pip install pyodbc
     # Also install ODBC Driver 17 for SQL Server from Microsoft
     ```
   
   - **Oracle**:
     ```bash
     pip install cx_oracle
     # Also install Oracle Instant Client
     ```

## Usage

1. Start Ollama:
```bash
ollama serve
```

2. Launch the application:
```bash
python run_streamlit.py
```

3. Access the web interface at `http://localhost:8501`

## Using the Chatbot

1. **Upload Documents**:
   - Use the file uploader in the sidebar
   - Supported formats: PDF, TXT, CSV, XLSX, DOC(X), PPT(X)
   - Multiple files can be uploaded simultaneously

2. **Select Model**:
   - Choose your preferred Ollama model from the dropdown
   - The model must be previously downloaded using `ollama pull`

3. **Start Chatting**:
   - Ask questions about your documents
   - Query your database using natural language
   - View sources and references in responses

4. **Connect to Database (Optional)**:
   - Use the sidebar to configure database connection
   - Provide host, port, database name, and credentials
   - Test connection and view schema

## Security Considerations

- All processing happens locally
- No data leaves your environment
- File validation and sanitization implemented
- Database connections use connection pooling
- Input sanitization for all user inputs
- Maximum file size limit: 50MB

## System Requirements

- Minimum 8GB RAM (16GB recommended)
- Multi-core CPU
- SSD storage for better vector search performance
- Network access for database connections (if using)

## Limitations

- File size limited to 50MB per file
- Performance depends on local hardware
- Currently supports one active database connection at a time
- Limited to Ollama-compatible models

## Troubleshooting

1. **Ollama Connection Issues**:
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/version
   ```

2. **Database Connection Issues**:
   - Verify database server is running
   - Check network connectivity
   - Confirm credentials and permissions
   - Verify required drivers are installed

3. **Memory Issues**:
   - Reduce chunk size in `rag_chatbot.py`
   - Process fewer documents simultaneously
   - Use a lighter LLM model

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://github.com/hwchase17/langchain)
- Vector store powered by [LanceDB](https://github.com/lancedb/lancedb)
- UI created with [Streamlit](https://streamlit.io)
- LLM support by [Ollama](https://ollama.ai)
