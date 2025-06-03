# examples/rag_cli.py

import os
import sys
import shutil
from typing import Optional

# --- Add project root to Python path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
# --- End path modification ---

try:
    from my_local_assistant.core.llm_handler import LLMHandler
    from my_local_assistant.core.vector_store import VectorStoreHandler
    from my_local_assistant.core.assistant import RAGAssistant
except ImportError as e:
    print(f"Failed to import necessary modules: {e}")
    print("Ensure your project structure is correct, you are in the 'examples' directory,")
    print("or that 'my_local_assistant' is installed and accessible.")
    sys.exit(1)

# --- Configuration (User should customize these paths) ---
# Path to your downloaded GGUF LLM model file
MODEL_FILE_NAME = "mistral-7b-instruct-v0.2.Q2_K.gguf" # <--- !!! USER: CHANGE THIS IF NEEDED !!!
MODEL_PATH = os.path.join(project_root, "models", MODEL_FILE_NAME)

# Directory to store the ChromaDB vector store data
VECTOR_DB_PERSIST_DIR = os.path.join(project_root, "data", "rag_cli_chroma_db")

# Directory where user documents for the knowledge base are stored
KNOWLEDGE_BASE_DOCS_DIR = os.path.join(project_root, "data", "knowledge_base_documents")
# --- End Configuration ---

def setup_directories():
    """Creates necessary directories if they don't exist."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) # models dir
    os.makedirs(VECTOR_DB_PERSIST_DIR, exist_ok=True)
    os.makedirs(KNOWLEDGE_BASE_DOCS_DIR, exist_ok=True)
    print("Checked/created necessary directories.")
    print(f"  LLM Model expected at: {MODEL_PATH}")
    print(f"  Vector DB will be stored in: {VECTOR_DB_PERSIST_DIR}")
    print(f"  Place your documents to index in: {KNOWLEDGE_BASE_DOCS_DIR}")

def initialize_assistant() -> Optional[RAGAssistant]:
    """Initializes and returns the RAGAssistant instance."""
    if not os.path.exists(MODEL_PATH):
        print(f"\nFATAL ERROR: LLM Model file not found at '{MODEL_PATH}'.")
        print("Please download a GGUF model, place it in the 'models' directory,")
        print(f"and ensure MODEL_FILE_NAME in '{__file__}' is correctly set.")
        return None

    print("\nInitializing services...")
    try:
        llm_h = LLMHandler(
            model_path=MODEL_PATH,
            n_ctx=2048,         # Adjust as needed for your model
            n_gpu_layers=0,     # Set to 0 for CPU-only
            verbose=False
        )
        print("LLM Handler initialized.")

        vs_h = VectorStoreHandler(
            collection_name="rag_cli_main_collection",
            persist_directory=VECTOR_DB_PERSIST_DIR,
            embedding_model_name='all-MiniLM-L6-v2' # Ensure this model is suitable
        )
        print(f"Vector Store Handler initialized. Knowledge base items: {vs_h.get_collection_count()}")

        assistant = RAGAssistant(llm_handler=llm_h, vector_store_handler=vs_h)
        assistant.set_persona("My Personal AI") # Customize your assistant's persona
        print("RAG Assistant initialized.")
        return assistant
    except Exception as e:
        print(f"Error during initialization: {e}")
        return None

def index_documents(assistant: RAGAssistant, docs_dir: str):
    """Scans a directory and indexes new/updated documents."""
    print(f"\nChecking for documents to index in: {docs_dir}...")
    if not os.path.isdir(docs_dir):
        print(f"Error: Document directory '{docs_dir}' not found.")
        return

    indexed_count = 0
    for filename in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, filename)
        if os.path.isfile(file_path) and (filename.lower().endswith((".txt", ".pdf", ".docx"))):
            # Simple check: you might want a more sophisticated way to track already indexed docs
            # For now, we re-index all, which is fine for small numbers of docs
            # or if ChromaDB handles deduplication or updates (it does by ID, but we generate new IDs each time here)
            # A more robust solution would involve storing a hash or timestamp of indexed files.
            print(f"  Indexing '{filename}'...")
            try:
                assistant.add_document_to_knowledge_base(
                    file_path,
                    chunk_size=700, # Adjust based on your content and model
                    chunk_overlap=150,
                    doc_metadata={"source_filename": filename}
                )
                indexed_count +=1
            except Exception as e:
                print(f"    Error indexing '{filename}': {e}")
        elif os.path.isfile(file_path):
            print(f"  Skipping '{filename}' (unsupported extension). Supported: .txt, .pdf, .docx")

    if indexed_count == 0:
        print("No new documents found or indexed from the directory.")
    else:
        print(f"Finished indexing. {indexed_count} file(s) processed.")


def main_cli_loop(assistant: RAGAssistant):
    """Main interactive loop for the CLI."""
    print("\n--- RAG CLI Assistant ---")
    print("Type your questions or commands.")
    print("Commands: '/quit', '/exit', '/index', '/clear_kb', '/help'")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/quit", "/exit"]:
                print("Exiting RAG CLI Assistant...")
                break
            elif user_input.lower() == "/help":
                print("\nAvailable commands:")
                print("  /quit, /exit         - Exit the assistant.")
                print("  /index               - Re-scan and index documents from the knowledge base directory.")
                print("  /clear_kb            - Clear all documents from the knowledge base (requires confirmation).")
                print("  /help                - Show this help message.")
                print("Any other input will be treated as a question to the assistant.")
            elif user_input.lower() == "/index":
                index_documents(assistant, KNOWLEDGE_BASE_DOCS_DIR)
            elif user_input.lower() == "/clear_kb":
                confirm = input("Are you sure you want to clear the entire knowledge base? (yes/no): ").lower()
                if confirm == 'yes':
                    assistant.clear_knowledge_base()
                else:
                    print("Knowledge base clear aborted.")
            else:
                print("Assistant thinking...")
                response = assistant.generate_rag_response(
                    query=user_input,
                    n_context_results=3, # Number of context chunks to retrieve
                    # llm_max_tokens=150 # Optionally override LLMHandler defaults
                )
                print(f"Assistant: {response}")

        except KeyboardInterrupt:
            print("\nExiting via KeyboardInterrupt...")
            break
        except Exception as e:
            print(f"\nAn error occurred in the main loop: {e}")
            # Optionally, re-initialize or offer to quit
            # For simplicity here, we just continue the loop.


def main_cli_loop_entry(): # <--- NEW FUNCTION TO BE THE ENTRY POINT
    """Main entry point for the CLI application."""
    setup_directories()
    rag_assistant_instance = initialize_assistant()

    if rag_assistant_instance:
        # Initial indexing of documents when CLI starts
        index_documents(rag_assistant_instance, KNOWLEDGE_BASE_DOCS_DIR)
        main_cli_loop(rag_assistant_instance)
    else:
        print("\nFailed to initialize the RAG Assistant. Exiting.")

if __name__ == "__main__":
    main_cli_loop_entry() # Call the new entry function