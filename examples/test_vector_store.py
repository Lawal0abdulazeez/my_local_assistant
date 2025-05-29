# examples/test_vector_store.py
import os
import sys
import shutil # For cleaning up test directories

# --- Add project root to Python path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
# --- End path modification ---

try:
    from my_local_assistant.core.vector_store import VectorStoreHandler, load_text_from_file
except ImportError as e:
    print(f"Failed to import VectorStoreHandler: {e}")
    print("Make sure your project structure is correct and you are in the 'examples' directory,")
    print("or that 'my_local_assistant' is installed.")
    sys.exit(1)

def create_dummy_files(base_path="."):
    """Creates dummy files for testing."""
    os.makedirs(base_path, exist_ok=True)
    dummy_files_info = {
        "notes_on_ai.txt": "Artificial intelligence is a rapidly evolving field. Machine learning is a subset of AI. Deep learning powers many modern AI applications.",
        "project_alpha_ideas.txt": "Project Alpha aims to improve customer satisfaction. Key ideas include a new feedback system and personalized recommendations. The target launch is Q4.",
        "random_thoughts.txt": "The weather is nice today. I should remember to buy groceries. What is the meaning of life?"
    }
    file_paths = []
    for filename, content in dummy_files_info.items():
        path = os.path.join(base_path, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        file_paths.append(path)
    print(f"Created dummy files in {base_path}")
    return file_paths

def cleanup_dummy_files_and_dir(file_paths, dir_path):
    """Removes dummy files and the directory."""
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    print(f"Cleaned up dummy files and directory: {dir_path}")


def run_vector_store_tests():
    print("--- Starting Vector Store Test Script ---")

    # Define paths for test data and database
    test_data_dir = os.path.join(project_root, "examples", "test_data_for_vector_store")
    test_db_persist_dir = os.path.join(project_root, "examples", "test_chroma_db")

    # Cleanup any previous test runs
    if os.path.exists(test_data_dir):
        shutil.rmtree(test_data_dir)
    if os.path.exists(test_db_persist_dir):
        shutil.rmtree(test_db_persist_dir)

    # Create dummy files for testing
    dummy_files = create_dummy_files(test_data_dir)

    # 1. Initialize VectorStoreHandler
    print("\n--- 1. Initializing VectorStoreHandler ---")
    vs_handler = VectorStoreHandler(
        collection_name="example_test_collection",
        persist_directory=test_db_persist_dir,
        embedding_model_name='all-MiniLM-L6-v2' # or 'paraphrase-MiniLM-L6-v2'
    )
    print(f"Initial collection count: {vs_handler.get_collection_count()}")

    # 2. Add documents from files
    print("\n--- 2. Adding Documents from Files ---")
    for file_path in dummy_files:
        vs_handler.add_document_from_file(
            file_path,
            chunk_size=100, # Smaller chunk size for these short docs
            chunk_overlap=10,
            doc_metadata={"source_type": "test_document"}
        )
    print(f"Collection count after adding documents: {vs_handler.get_collection_count()}")

    # 3. Perform Queries
    print("\n--- 3. Performing Queries ---")
    queries = {
        "AI applications": "Tell me about AI applications.",
        "Project Alpha details": "What are the key ideas for Project Alpha?",
        "Groceries reminder": "What should I remember about groceries?",
        "Non-existent topic": "Tell me about space travel." # Should yield less relevant results
    }

    for desc, query_text in queries.items():
        print(f"\n--- Querying for: '{desc}' ---")
        results = vs_handler.query(query_text, n_results=2)
        if results and results.get('documents') and results['documents'][0]:
            for i, doc_text in enumerate(results['documents'][0]):
                print(f"  Result {i+1} (Distance: {results['distances'][0][i]:.4f}):")
                print(f"    ID: {results['ids'][0][i]}")
                print(f"    Metadata: {results['metadatas'][0][i]}")
                print(f"    Text: \"{doc_text[:150].strip()}...\"")
        else:
            print(f"  No relevant results found for '{desc}' or an error occurred.")

    # 4. Test Query with Metadata Filter
    print("\n--- 4. Querying with Metadata Filter (source_type='test_document') ---")
    filtered_query = "machine learning"
    filter_metadata = {"source_type": "test_document"}
    results_filtered = vs_handler.query(filtered_query, n_results=1, query_filter=filter_metadata)
    if results_filtered and results_filtered.get('documents') and results_filtered['documents'][0]:
        doc_text = results_filtered['documents'][0][0]
        meta = results_filtered['metadatas'][0][0]
        print(f"  Filtered Result for '{filtered_query}':")
        print(f"    Metadata: {meta}")
        print(f"    Text: \"{doc_text[:150].strip()}...\"")
    else:
        print(f"  No results for '{filtered_query}' with filter or error.")


    # 5. Test Resetting the Collection
    print("\n--- 5. Resetting the Collection ---")
    vs_handler.reset_collection()
    print(f"Collection count after reset: {vs_handler.get_collection_count()}")

    # Cleanup
    print("\n--- Cleaning up test files and directory ---")
    cleanup_dummy_files_and_dir(dummy_files, test_data_dir)
    # The ChromaDB directory is also cleaned up as part of the test run,
    # or you can explicitly clean test_db_persist_dir if you want to ensure a fresh start next time.
    if os.path.exists(test_db_persist_dir):
        shutil.rmtree(test_db_persist_dir)
    print(f"Cleaned up ChromaDB test directory: {test_db_persist_dir}")

    print("\n--- Vector Store Test Script Complete ---")

if __name__ == "__main__":
    run_vector_store_tests()