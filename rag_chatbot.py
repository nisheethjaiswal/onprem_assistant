# Part 1: Imports and Setup
from datetime import datetime
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_sql_query_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.schema.document import Document
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import SQLDatabase
try:
    from langchain_ollama.llms import Ollama as OllamaLLM
except ImportError:
    from langchain_ollama import OllamaLLM  # fallback for older versions
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import QueuePool
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Optional, Dict, Tuple, Any
import hashlib
import json
import lancedb
import logging
import os
import platform
import pyarrow as pa
import requests
import streamlit as st
import sys
import tempfile
import urllib.parse

# Database-specific imports (install as needed)
try:
    import psycopg2  # for PostgreSQL
except ImportError:
    pass

try:
    import pymysql  # for MySQL
except ImportError:
    pass

try:
    import pyodbc  # for MS SQL Server
except ImportError:
    pass

try:
    import cx_Oracle  # for Oracle
except ImportError:
    pass

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_RETRIES = 3
BASE_TIMEOUT = 10
OLLAMA_API_URL = "http://localhost:11434/api"
PERSIST_DIR = Path("./persistent_storage")
TEMP_DIR = Path("./temp_storage")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable CUDA to avoid DLL errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Part 2: Security Manager Class
class SecurityManager:
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input by removing leading/trailing whitespace"""
        return text.strip()

    @staticmethod
    def validate_file(file) -> bool:
        """
        Validate uploaded file size and type
        
        Args:
            file: StreamlitUploadedFile object
            
        Returns:
            bool: True if file is valid
            
        Raises:
            ValueError: If file size exceeds limit or file type is not supported
        """
        if file.size > MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds limit of {MAX_FILE_SIZE/1024/1024}MB")
        
        allowed_extensions = {'.pdf', '.txt', '.csv', '.xlsx', '.xls', 
                            '.doc', '.docx', '.ppt', '.pptx'}
        ext = Path(file.name).suffix.lower()
        
        if ext not in allowed_extensions:
            raise ValueError(f"Unsupported file type: {ext}")
            
        return True

    @staticmethod
    def get_file_hash(file) -> str:
        """
        Generate SHA-256 hash for file content
        
        Args:
            file: StreamlitUploadedFile object
            
        Returns:
            str: Hexadecimal hash of file content
        """
        return hashlib.sha256(file.getvalue()).hexdigest()
# Part 3: Document Processor
class DocumentProcessor:
    def __init__(self):
        """Initialize the DocumentProcessor with text splitter and embeddings"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        # Initialize HuggingFaceEmbeddings with a small, efficient model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.security = SecurityManager()
        
        # Ensure directories exist
        PERSIST_DIR.mkdir(exist_ok=True)
        TEMP_DIR.mkdir(exist_ok=True)

    def process_file(self, file) -> Tuple[Optional[List[Dict]], str]:
        """
        Process uploaded file and generate embeddings
        
        Args:
            file: StreamlitUploadedFile object
            
        Returns:
            Tuple containing:
                - List of dictionaries with text, metadata, and vector embeddings
                - File hash string
            
        Raises:
            Exception: If file processing fails
        """
        try:
            self.security.validate_file(file)
            file_hash = self.security.get_file_hash(file)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(dir=TEMP_DIR, delete=False) as temp_file:
                temp_file.write(file.getvalue())
                temp_path = Path(temp_file.name)
                
            try:
                documents = self._load_file(temp_path, file.name)
                
                # Convert documents to the format expected by LanceDB
                texts = []
                for doc in documents:
                    texts.append({
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "vector": self.embeddings.embed_query(doc.page_content)
                    })
                    
                return texts, file_hash
                
            finally:
                # Clean up temporary file
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            raise

    def _load_file(self, file_path: Path, original_filename: str) -> List[str]:
        """
        Load file content based on file extension
        
        Args:
            file_path: Path to temporary file
            original_filename: Original uploaded filename
            
        Returns:
            List of document chunks
            
        Raises:
            ValueError: If no loader available for file type
        """
        ext = Path(original_filename).suffix.lower()
        loader_class = self._get_loader_class(ext)
        
        if not loader_class:
            raise ValueError(f"No loader available for {ext}")
            
        loader = loader_class(str(file_path))
        documents = loader.load()
        
        return self.text_splitter.split_documents(documents)

    @staticmethod
    def _get_loader_class(ext: str):
        """Get appropriate document loader class based on file extension"""
        loaders = {
            '.pdf': PyPDFLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            '.csv': CSVLoader,
            '.txt': TextLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.ppt': UnstructuredPowerPointLoader,
            '.pptx': UnstructuredPowerPointLoader
        }
        return loaders.get(ext)
# Part 4: LanceDB Implementation
class LanceDBStore(VectorStore):
    def __init__(self, table, embedding):
        """Initialize LanceDB store with table and embeddings"""
        super().__init__()
        self.table = table
        self._embedding = embedding
        self.k = 3  # Default number of results to return
        
    @property
    def embedding(self):
        return self._embedding
    
    @embedding.setter
    def embedding(self, value):
        self._embedding = value

    def add_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]] = None, **kwargs
    ) -> List[str]:
        """
        Add texts to the vectorstore
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dicts for each text
            **kwargs: Additional arguments
            
        Returns:
            List[str]: List of IDs for added texts
        """
        if not metadatas:
            metadatas = [{} for _ in texts]
        
        vectors = [self.embedding.embed_query(text) for text in texts]
        processed_texts = []
        for text, metadata, vector in zip(texts, metadatas, vectors):
            processed_texts.append({
                "text": text,
                "vector": vector,
                "metadata": str(metadata)
            })
        self.table.add(processed_texts)
        return [str(i) for i in range(len(texts))]

    def similarity_search(
        self, query: str, k: int = 3, **kwargs
    ) -> List[Document]:
        """
        Run similarity search with query
        
        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List[Document]: List of similar documents
        """
        query_vector = self.embedding.embed_query(query)
        results = self.table.search(query_vector).limit(k).to_list()
        
        documents = []
        for result in results:
            import ast
            metadata = ast.literal_eval(result['metadata'])
            documents.append(Document(
                page_content=result['text'],
                metadata=metadata
            ))
        return documents

    async def asimilarity_search(
        self, query: str, k: int = 3, **kwargs
    ) -> List[Document]:
        """Async version of similarity search"""
        return self.similarity_search(query, k, **kwargs)
        
    def save_local(self, folder_path: str, index_name: str) -> None:
        """Save vectorstore to disk - Not implemented for LanceDB"""
        pass  # LanceDB handles persistence automatically
        
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        """Create LanceDBStore from texts - Not implemented"""
        raise NotImplementedError(
            "LanceDBStore does not support from_texts construction."
        )

class LanceDBManager:
    def __init__(self):
        """Initialize LanceDB connection and embeddings"""
        self.db = lancedb.connect(str(PERSIST_DIR / "lancedb"))
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_or_get_table(self, texts: List[Dict], table_name: str):
        """
        Create or get existing LanceDB table and return as retriever
        
        Args:
            texts: List of dictionaries containing text, metadata, and vectors
            table_name: Name of the table to create or retrieve
            
        Returns:
            VectorStoreRetriever: Retriever interface for the table
            
        Raises:
            Exception: If table creation or retrieval fails
        """
        try:
            # Check if table exists
            if table_name in self.db.table_names():
                table = self.db.open_table(table_name)
            else:
                # Create schema using pyarrow
                schema = pa.schema([
                    ('text', pa.string()),
                    ('vector', pa.list_(pa.float32(), 384)),  # 384 is the dimension for all-MiniLM-L6-v2
                    ('metadata', pa.string())
                ])
                
                # Convert metadata to string for storage
                processed_texts = []
                for text in texts:
                    processed_texts.append({
                        "text": text["text"],
                        "vector": text["vector"],
                        "metadata": str(text["metadata"])
                    })
                    
                table = self.db.create_table(table_name, schema=schema, mode="overwrite")
                table.add(processed_texts)
            
            # Create vectorstore and wrap in retriever
            vectorstore = LanceDBStore(table, embedding=self.embeddings)
            return VectorStoreRetriever(vectorstore=vectorstore, search_kwargs={"k": 3})
            
        except Exception as e:
            logger.error(f"Error with LanceDB: {e}")
            raise
class DatabaseManager:
    DATABASE_CONFIGS = {
        "postgresql": {
            "default_port": 5432,
            "driver": "postgresql",
            "display_name": "PostgreSQL",
            "connection_format": "postgresql://{username}:{password}@{host}:{port}/{database}"
        },
        "mysql": {
            "default_port": 3306,
            "driver": "mysql+pymysql",
            "display_name": "MySQL",
            "connection_format": "mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        },
        "mssql": {
            "default_port": 1433,
            "driver": "mssql+pyodbc",
            "display_name": "Microsoft SQL Server",
            "connection_format": "mssql+pyodbc://{username}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
        },
        "oracle": {
            "default_port": 1521,
            "driver": "oracle+cx_oracle",
            "display_name": "Oracle",
            "connection_format": "oracle+cx_oracle://{username}:{password}@{host}:{port}/?service_name={database}"
        }
    }

    def __init__(self):
        """Initialize database manager"""
        self.db_uri = None
        self.engine = None
        self.db_chain = None
        self.current_config = None

    def format_connection_string(self, config: dict) -> str:
        """Format connection string using the provided configuration"""
        # URL encode the password to handle special characters
        config['password'] = urllib.parse.quote_plus(config['password'])
        return config['format'].format(**config)

    def connect_database(
        self,
        db_type: str,
        host: str,
        port: str,
        database: str,
        username: str,
        password: str,
        **kwargs
    ) -> bool:
        """
        Connect to database with given parameters
        
        Args:
            db_type: Type of database (postgresql, mysql, etc.)
            host: Database server hostname
            port: Port number
            database: Database name
            username: Database username
            password: Database password
            **kwargs: Additional connection parameters
        """
        try:
            if db_type not in self.DATABASE_CONFIGS:
                raise ValueError(f"Unsupported database type: {db_type}")

            # Prepare connection configuration
            config = {
                'format': self.DATABASE_CONFIGS[db_type]['connection_format'],
                'host': host,
                'port': port,
                'database': database,
                'username': username,
                'password': password
            }

            # Create connection string
            self.db_uri = self.format_connection_string(config)
            self.current_config = {
                'type': db_type,
                'host': host,
                'port': port,
                'database': database,
                'username': username
            }

            # Initialize engine with connection pooling
            self.engine = create_engine(
                self.db_uri,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800
            )

            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info(f"Successfully connected to {db_type} database at {host}:{port}")
            return True

        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    def get_connection_info(self) -> str:
        """Get formatted connection information"""
        if not self.current_config:
            return "Not connected to any database"

        db_type = self.current_config['type']
        display_name = self.DATABASE_CONFIGS[db_type]['display_name']
        
        return (
            f"Connected to {display_name}\n"
            f"Host: {self.current_config['host']}\n"
            f"Port: {self.current_config['port']}\n"
            f"Database: {self.current_config['database']}\n"
            f"Username: {self.current_config['username']}"
        )

    def get_table_info(self) -> str:
        """Get information about database tables and their columns"""
        if not self.engine:
            return "No active database connection"

        try:
            inspector = inspect(self.engine)
            table_info = []

            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                col_info = [f"{col['name']} ({col['type']})" for col in columns]
                pk_columns = inspector.get_pk_constraint(table_name)['constrained_columns']
                
                table_info.append(
                    f"Table: {table_name}\n"
                    f"Primary Keys: {', '.join(pk_columns) if pk_columns else 'None'}\n"
                    f"Columns: {', '.join(col_info)}\n"
                )

            return "\n".join(table_info) if table_info else "No tables found"

        except Exception as e:
            logger.error(f"Error getting table information: {e}")
            raise

    def execute_query(self, query: str) -> str:
        """Execute SQL query and return results as markdown table"""
        if not self.engine:
            raise ValueError("No active database connection")

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()

                if not rows:
                    return "Query executed successfully. No results to display."

                # Format as markdown table
                headers = result.keys()
                table = "| " + " | ".join(headers) + " |\n"
                table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                
                for row in rows:
                    formatted_row = [str(cell) if cell is not None else "NULL" for cell in row]
                    table += "| " + " | ".join(formatted_row) + " |\n"

                return table

        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise

    def initialize_nl2sql_chain(self, llm) -> bool:
        """Initialize natural language to SQL chain"""
        if not self.db_uri:
            raise ValueError("Database connection required for NL2SQL chain")

        try:
            db = SQLDatabase.from_uri(self.db_uri)
            self.db_chain = create_sql_query_chain(llm, db)
            logger.info("Successfully initialized NL2SQL chain")
            return True

        except Exception as e:
            logger.error(f"Error initializing NL2SQL chain: {e}")
            raise
# Part 6: Chatbot Manager
class ChatbotManager:
    def __init__(self):
        """Initialize chatbot manager with security and database managers"""
        self.security = SecurityManager()
        self.db_manager = DatabaseManager()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def get_available_models(self) -> List[str]:
        """
        Get list of available Ollama models with retry logic
        
        Returns:
            List[str]: List of available model names
            
        Raises:
            Exception: If fetching models fails after retries
        """
        try:
            response = requests.get(
                f"{OLLAMA_API_URL}/tags", 
                timeout=BASE_TIMEOUT
            )
            response.raise_for_status()
            
            models_data = response.json()
            available_models = [model['name'] for model in models_data['models']]
            logger.info(f"Available models: {available_models}")
            
            return available_models
            
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            # Return a default model list if Ollama is not available
            default_models = ["llama2"]
            logger.info(f"Using default models: {default_models}")
            return default_models

    def initialize_chain(self, retriever: VectorStoreRetriever, model_name: str):
        """
        Initialize conversation chain with RAG and optional NL2SQL capabilities
        
        Args:
            retriever: Document retriever instance
            model_name: Name of the language model to use
            
        Returns:
            ConversationalRetrievalChain: Initialized conversation chain
            
        Raises:
            Exception: If chain initialization fails
        """
        try:
            llm = OllamaLLM(
                model=model_name,
                temperature=0.7,
                base_url=OLLAMA_API_URL.replace('/api', '')
            )
            
            # Initialize RAG chain using the retriever directly
            rag_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )
            
            # Initialize NL2SQL chain if database is connected
            if self.db_manager.db_uri:
                self.db_manager.initialize_nl2sql_chain(llm)
                
            logger.info(f"Successfully initialized chain with model: {model_name}")
            return rag_chain
            
        except Exception as e:
            logger.error(f"Error initializing chain: {e}")
            raise

    def process_query(
        self, 
        query: str, 
        conversation_chain: ConversationalRetrievalChain, 
        chat_history: List[Tuple[str, str]] = None
    ) -> str:
        """
        Process user query using appropriate chain (RAG or NL2SQL)
        
        Args:
            query: User input query
            conversation_chain: Active conversation chain
            chat_history: Optional list of previous conversation turns
            
        Returns:
            str: Formatted response with sources or query results
            
        Raises:
            Exception: If query processing fails
        """
        try:
            # Handle database query if relevant
            if "database" in query.lower() and self.db_manager.db_uri:
                sql_query = self.db_manager.db_chain.invoke(query)
                results = self.db_manager.execute_query(sql_query)
                return (
                    f"Generated SQL:\n```sql\n{sql_query}\n```\n\n"
                    f"Results:\n{results}"
                )
                
            # Handle RAG query
            response = conversation_chain({
                "question": query,
                "chat_history": chat_history or []
            })
            
            # Format response with sources
            answer = response["answer"]
            if "source_documents" in response and response["source_documents"]:
                answer += "\n\nSources:\n"
                for i, doc in enumerate(response["source_documents"], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    answer += f"{i}. Source: {source}, Page: {page}\n"
                    
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
# Part 7: Main Application
def initialize_session_state():
    """Initialize Streamlit session state with required components"""
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
        
    if 'lancedb_manager' not in st.session_state:
        st.session_state.lancedb_manager = LanceDBManager()
        
    if 'chatbot_manager' not in st.session_state:
        st.session_state.chatbot_manager = ChatbotManager()
        
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None

def display_sidebar() -> Optional[str]:
    """Configure and display sidebar elements"""
    with st.sidebar:
        # System Information
        st.write("System Information:")
        st.write(f"- OS: {platform.system()} {platform.release()}")
        st.write(f"- Python: {sys.version.split()[0]}")
        
        # Database Connection Section
        st.header("Database Connection")
        
        # Show current connection if exists
        db_manager = st.session_state.chatbot_manager.db_manager
        if db_manager.current_config:
            with st.expander("Current Connection", expanded=True):
                st.info(db_manager.get_connection_info())
        
        # Database Connection Form
        with st.form("db_connection_form"):
            st.subheader("New Connection")
            
            # Database Type Selection
            db_type = st.selectbox(
                "Database Type",
                options=list(DatabaseManager.DATABASE_CONFIGS.keys()),
                format_func=lambda x: DatabaseManager.DATABASE_CONFIGS[x]['display_name']
            )
            
            # Connection Details
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input(
                    "Host",
                    value="localhost",
                    help="Database server address"
                )
            with col2:
                port = st.text_input(
                    "Port",
                    value=str(DatabaseManager.DATABASE_CONFIGS[db_type]['default_port']),
                    help="Database port number"
                )
                
            # Database name
            database = st.text_input(
                "Database Name",
                help="Name of the database to connect to"
            )
            
            # Authentication
            col3, col4 = st.columns(2)
            with col3:
                username = st.text_input(
                    "Username",
                    help="Database user account"
                )
            with col4:
                password = st.text_input(
                    "Password",
                    type="password",
                    help="Database user password"
                )
            
            # Help information for different database types
            help_texts = {
                "postgresql": """
                    - Ensure PostgreSQL server is running
                    - Check pg_hba.conf for authentication settings
                    - Verify network connectivity to the server
                """,
                "mysql": """
                    - Verify MySQL server is running
                    - Check user privileges
                    - Ensure bind-address configuration allows remote connections
                """,
                "mssql": """
                    - SQL Server must have TCP/IP protocol enabled
                    - SQL Server Authentication mode must be enabled
                    - Appropriate ODBC driver must be installed
                """,
                "oracle": """
                    - Oracle listener must be running
                    - Database service name must be correctly configured
                    - TNS configuration must be properly set up
                """
            }
            
            with st.expander("Connection Help", expanded=False):
                st.markdown(help_texts[db_type])
            
            # Connect Button
            submitted = st.form_submit_button("Connect to Database")
            if submitted:
                try:
                    with st.spinner("Connecting to database..."):
                        if db_manager.connect_database(
                            db_type=db_type,
                            host=host,
                            port=port,
                            database=database,
                            username=username,
                            password=password
                        ):
                            st.success("Successfully connected to database!")
                            
                            # Show database schema in expander
                            with st.expander("Database Schema", expanded=True):
                                table_info = db_manager.get_table_info()
                                st.code(table_info)
                                
                except Exception as e:
                    st.error(f"Failed to connect to database: {str(e)}")
                    logging.error(f"Database connection error: {e}", exc_info=True)
        
        st.markdown("---")
        
        # Model Selection Section
        st.header("Model Selection")
        try:
            available_models = st.session_state.chatbot_manager.get_available_models()
            selected_model = st.selectbox(
                "Select Model",
                options=available_models,
                help="Choose the AI model for processing queries"
            )
            return selected_model
            
        except Exception as e:
            st.error("Error connecting to Ollama. Please ensure it's running.")
            st.error(f"Error details: {str(e)}")
            return None
def handle_file_upload(selected_model: Optional[str]):
    """
    Handle file upload and processing
    
    Args:
        selected_model: Name of selected language model
    """
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'csv', 'xlsx', 'doc', 'docx', 'ppt', 'pptx']
    )
    
    if uploaded_files and selected_model:
        try:
            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    texts, file_hash = st.session_state.processor.process_file(file)
                    if texts:
                        table_name = f"table_{file_hash}"
                        retriever = st.session_state.lancedb_manager.create_or_get_table(
                            texts, 
                            table_name
                        )
                        st.session_state.conversation_chain = (
                            st.session_state.chatbot_manager.initialize_chain(
                                retriever, 
                                selected_model
                            )
                        )
                st.success("Documents processed successfully!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            logger.error(f"Document processing error: {e}", exc_info=True)

def display_chat_interface():
    """Display and handle chat interface"""
    st.header("Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents or database"):
        prompt = SecurityManager.sanitize_input(prompt)
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            if not st.session_state.conversation_chain:
                st.error("Please upload documents and select a model first!")
            else:
                try:
                    with st.spinner("Thinking..."):
                        # Get chat history
                        chat_history = [
                            (m["content"], r["content"]) 
                            for m, r in zip(
                                st.session_state.messages[::2], 
                                st.session_state.messages[1::2]
                            )
                        ]
                        
                        # Process query
                        response = st.session_state.chatbot_manager.process_query(
                            prompt,
                            st.session_state.conversation_chain,
                            chat_history
                        )
                        
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                except Exception as e:
                    error_msg = f"Error processing your question: {str(e)}"
                    logger.error(f"Query processing error: {e}", exc_info=True)
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

def main():
    """Main application entry point"""
    # Set environment variable to disable CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="KC",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 KC")
    
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar and get selected model
    selected_model = display_sidebar()
    
    # Handle file upload
    handle_file_upload(selected_model)
    
    # Display chat interface
    display_chat_interface()

if __name__ == "__main__":
    main()


