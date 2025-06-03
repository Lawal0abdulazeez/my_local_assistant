# examples/rag_cli.py # Or, more accurately if run with `python -m my_local_assistant.cli.rag_cli`,
# this file would be something like: my_local_assistant/cli/rag_cli.py

import os
import sys # Keep sys for sys.exit
import shutil # shutil is not used, can be removed if not planned for future use
from typing import Optional
from pathlib import Path # Keep Path for path manipulations

# --- Add project root to Python path ---
# This block is NOT needed when running with `python -m my_local_assistant.cli.rag_cli`
# as Python handles the path for modules correctly.
# If you were running `python examples/rag_cli.py` directly from the project root,
# then such a block (or adding the project root to PYTHONPATH) would be necessary.
# For `python -m ...` execution, it's best to remove it.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, ".."))
# sys.path.insert(0, project_root)
# --- End path modification ---


try:
    # These relative imports are correct for `python -m my_local_assistant.cli.rag_cli`
    # assuming `core` is a sibling directory to `cli` under `my_local_assistant` package
    from ..core.llm_handler import LLMHandler
    from ..core.vector_store import VectorStoreHandler
    from ..core.assistant import RAGAssistant
except ImportError as e:
    print(f"Failed to import necessary modules: {e}")
    print("Ensure your project structure is correct (e.g., 'my_local_assistant' package with 'core' and 'cli' submodules),")
    print("and you are running with `python -m my_local_assistant.cli.rag_cli` from the project root directory,")
    print("or that 'my_local_assistant' is installed and accessible in your Python environment.")
    sys.exit(1)


# Your existing path definitions - these are fine and create paths like
# C:\Users\YOUR_USER\Desktop\my_local_assistant\models
# which matches your initial error message's expectation.
APP_NAME = "Desktop\\my_local_assistant" # Using double backslash or raw string for Windows paths in strings is safer
# Or better, rely on Path's / operator: APP_DIR_NAME = "my_local_assistant"
# USER_DATA_DIR = Path.home() / "Desktop" / APP_DIR_NAME
# However, your current APP_NAME leads to the paths mentioned in your setup instructions.
USER_DATA_DIR = Path.home() / APP_NAME
MODEL_DIR = USER_DATA_DIR / "models"
VECTOR_DB_PERSIST_DIR = USER_DATA_DIR / "rag_cli_chroma_db"
KNOWLEDGE_BASE_DOCS_DIR = USER_DATA_DIR / "knowledge_base_documents"

MODEL_FILE_NAME = "mistral-7b-instruct-v0.2.Q2_K.gguf"
MODEL_PATH = MODEL_DIR / MODEL_FILE_NAME # This is a Path object


def setup_directories():
    """Creates necessary directories in user's home if they don't exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_BASE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # The print statements are fine, using Path objects here is okay as they convert to string for printing.
    print(f"  Place your GGUF model in: {MODEL_DIR}")
    print(f"  Models like '{MODEL_FILE_NAME}' should be in {MODEL_DIR}")
    print(f"  Place documents for the knowledge base in: {KNOWLEDGE_BASE_DOCS_DIR}")
    print(f"  Vector database will be stored in: {VECTOR_DB_PERSIST_DIR}")


def initialize_assistant() -> Optional[RAGAssistant]:
    """Initializes and returns the RAGAssistant instance."""
    # os.path.exists works fine with Path objects
    if not MODEL_PATH.exists(): # Changed os.path.exists(MODEL_PATH) to MODEL_PATH.exists() for consistency with Path objects
        print(f"\nFATAL ERROR: LLM Model file not found at '{MODEL_PATH}'.")
        print("Please download a GGUF model, place it in the 'models' directory,")
        print(f"and ensure MODEL_FILE_NAME in '{__file__}' is correctly set.")
        return None

    print("\nInitializing services...")
    try:
        print(f"Loading LLM from: {MODEL_PATH}...") # For user feedback
        llm_h = LLMHandler(
            model_path=str(MODEL_PATH), # <<< CRITICAL FIX: Convert Path object to string
            n_ctx=2048,
            n_gpu_layers=0,
            verbose=False
        )
        print("LLM Handler initialized.")

        vs_h = VectorStoreHandler(
            collection_name="rag_cli_main_collection",
            # Convert to string for ChromaDB or similar libraries if they expect string paths
            persist_directory=str(VECTOR_DB_PERSIST_DIR), # <<< RECOMMENDED FIX
            embedding_model_name='all-MiniLM-L6-v2'
        )
        print(f"Vector Store Handler initialized. Knowledge base items: {vs_h.get_collection_count()}")

        assistant = RAGAssistant(llm_handler=llm_h, vector_store_handler=vs_h)
        assistant.set_persona("My Personal AI")
        print("RAG Assistant initialized.")
        return assistant
    except Exception as e:
        # It's often helpful to print the full traceback for debugging initialization errors
        import traceback
        print(f"Error during initialization: {e}")
        traceback.print_exc() # Prints the full stack trace
        return None

def index_documents(assistant: RAGAssistant, docs_dir: Path): # Changed type hint to Path
    """Scans a directory and indexes new/updated documents."""
    print(f"\nChecking for documents to index in: {docs_dir}...")
    if not docs_dir.is_dir(): # Changed os.path.isdir to docs_dir.is_dir()
        print(f"Error: Document directory '{docs_dir}' not found.")
        return

    indexed_count = 0
    # os.listdir works with Path objects, but Path.iterdir() is more idiomatic with pathlib
    for item in docs_dir.iterdir():
        if item.is_file() and item.suffix.lower() in [".txt", ".pdf", ".docx"]:
            print(f"  Indexing '{item.name}'...")
            try:
                # assistant.add_document_to_knowledge_base likely expects a string file path
                # If it's well-designed, it might handle Path objects too, but str() is safest.
                assistant.add_document_to_knowledge_base(
                    str(item), # Convert Path object to string
                    chunk_size=700,
                    chunk_overlap=150,
                    doc_metadata={"source_filename": item.name}
                )
                indexed_count +=1
            except Exception as e:
                print(f"    Error indexing '{item.name}': {e}")
        elif item.is_file():
            print(f"  Skipping '{item.name}' (unsupported extension). Supported: .txt, .pdf, .docx")

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
                index_documents(assistant, KNOWLEDGE_BASE_DOCS_DIR) # KNOWLEDGE_BASE_DOCS_DIR is a Path object
            elif user_input.lower() == "/clear_kb":
                confirm = input("Are you sure you want to clear the entire knowledge base? (yes/no): ").lower()
                if confirm == 'yes':
                    assistant.clear_knowledge_base()
                    print("Knowledge base cleared.")
                else:
                    print("Knowledge base clear aborted.")
            else:
                print("Assistant thinking...")
                response = assistant.generate_rag_response(
                    query=user_input,
                    n_context_results=3,
                )
                print(f"Assistant: {response}")

        except KeyboardInterrupt:
            print("\nExiting via KeyboardInterrupt...")
            break
        except Exception as e:
            print(f"\nAn error occurred in the main loop: {e}")
            # import traceback # Uncomment for debugging
            # traceback.print_exc() # Uncomment for debugging


def main_cli_loop_entry():
    """Main entry point for the CLI application."""
    setup_directories()
    rag_assistant_instance = initialize_assistant()

    if rag_assistant_instance:
        index_documents(rag_assistant_instance, KNOWLEDGE_BASE_DOCS_DIR)
        main_cli_loop(rag_assistant_instance)
    else:
        print("\nFailed to initialize the RAG Assistant. Exiting.")

if __name__ == "__main__":
    main_cli_loop_entry()