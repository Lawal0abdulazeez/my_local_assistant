# my_local_assistant/core/vector_store.py

import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import uuid # For generating unique IDs for chunks

# For more advanced text splitting, consider LangChain later
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Helper for loading different file types ---
def load_text_from_file(file_path: str) -> Optional[str]:
    """Loads text content from various file types."""
    try:
        if file_path.lower().endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.lower().endswith(".pdf"):
            try:
                import pypdf
                text = ""
                with open(file_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page_num in range(len(reader.pages)):
                        text += reader.pages[page_num].extract_text()
                return text
            except ImportError:
                print("pypdf not installed. Please run 'pip install pypdf' to support PDF files.")
                return None
            except Exception as e:
                print(f"Error reading PDF {file_path}: {e}")
                return None
        elif file_path.lower().endswith(".docx"):
            try:
                import docx
                doc = docx.Document(file_path)
                full_text = []
                for para in doc.paragraphs:
                    full_text.append(para.text)
                return '\n'.join(full_text)
            except ImportError:
                print("python-docx not installed. Please run 'pip install python-docx' to support .docx files.")
                return None
            except Exception as e:
                print(f"Error reading DOCX {file_path}: {e}")
                return None
        else:
            print(f"Unsupported file type: {file_path}. Only .txt, .pdf, .docx are currently supported.")
            return None
    except Exception as e:
        print(f"Error opening or reading file {file_path}: {e}")
        return None

class VectorStoreHandler:
    """
    Handles creating, populating, and querying a ChromaDB vector store
    using SentenceTransformer embeddings.
    """
    def __init__(self,
                 collection_name: str = "my_documents_collection",
                 persist_directory: str = "./db_chroma_data", # Path to store DB data
                 embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the VectorStoreHandler.

        Args:
            collection_name (str): Name of the ChromaDB collection.
            persist_directory (str): Directory to persist ChromaDB data.
            embedding_model_name (str): Name of the SentenceTransformer model to use.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name

        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        print(f"Initializing SentenceTransformer model: {self.embedding_model_name}...")
        # ChromaDB can use sentence-transformers directly if specified
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        print("SentenceTransformer model loaded.")

        print(f"Initializing ChromaDB client with persistence at: {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        print(f"Getting or creating Chroma collection: {self.collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.sentence_transformer_ef # Pass the embedding function here
        )
        print(f"Collection '{self.collection_name}' ready. Item count: {self.collection.count()}")


    def _simple_chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
        """
        A very simple text chunker.
        For more robust chunking, consider libraries like LangChain's text_splitters.
        This one splits by characters. Better would be by tokens or semantic units.
        """
        if not text:
            return []
        chunks = []
        start_index = 0
        while start_index < len(text):
            end_index = min(start_index + chunk_size, len(text))
            chunks.append(text[start_index:end_index])
            start_index += chunk_size - chunk_overlap
            if start_index >= len(text) and end_index < len(text) and chunk_overlap > 0 : # Ensure last part is captured if overlap makes us miss it
                if len(text[start_index - (chunk_size - chunk_overlap):]) > chunk_overlap : # if the remaining part is not too small
                     pass # it was already captured
                elif start_index - (chunk_size-chunk_overlap) + chunk_size < len(text): # check if a last chunk can be made
                     chunks.append(text[start_index - (chunk_size - chunk_overlap) + chunk_size - (chunk_size - chunk_overlap):])

        # A refinement to ensure the very last bit of text isn't missed if small
        if chunks and len(text) > (len(chunks) -1) * (chunk_size - chunk_overlap) + chunk_size :
             last_chunk_start = (len(chunks) -1) * (chunk_size - chunk_overlap)
             if len(text[last_chunk_start:]) > 0 :
                 chunks[-1] = text[last_chunk_start:] # Replace last chunk with everything remaining from its start

        return [chunk for chunk in chunks if chunk.strip()] # Remove empty chunks

    def add_text_batch(self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """
        Adds a batch of texts to the vector store.

        Args:
            texts (List[str]): A list of text strings to add.
            metadatas (List[Dict], optional): A list of metadata dictionaries, one for each text.
                                              Must be the same length as texts.
            ids (List[str], optional): A list of unique IDs for each text.
                                       Must be the same length as texts. If None, UUIDs are generated.
        """
        if not texts:
            print("No texts provided to add_text_batch.")
            return

        num_texts = len(texts)
        if metadatas and len(metadatas) != num_texts:
            raise ValueError("Length of texts and metadatas must match.")
        if ids and len(ids) != num_texts:
            raise ValueError("Length of texts and ids must match.")

        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(num_texts)]
        if not metadatas:
            metadatas = [{} for _ in range(num_texts)] # Default to empty metadata

        print(f"Adding {num_texts} text chunks to collection '{self.collection_name}'...")
        try:
            self.collection.add(
                documents=texts,    # In ChromaDB, 'documents' are the text content
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {num_texts} chunks. Collection count: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to Chroma: {e}")


    def add_document_from_file(self,
                               file_path: str,
                               chunk_size: int = 1000,
                               chunk_overlap: int = 100,
                               doc_metadata: Optional[Dict] = None):
        """
        Loads, chunks, and adds a document from a file path to the vector store.

        Args:
            file_path (str): Path to the document file (.txt, .pdf, .docx).
            chunk_size (int): Size of chunks for splitting the document.
            chunk_overlap (int): Overlap between chunks.
            doc_metadata (Optional[Dict]): Metadata to associate with all chunks from this document.
                                          'source' will be automatically added/overridden with file_path.
        """
        print(f"Processing document: {file_path}")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        text_content = load_text_from_file(file_path)
        if not text_content:
            print(f"Could not extract text from {file_path}")
            return

        chunks = self._simple_chunk_text(text_content, chunk_size, chunk_overlap)
        if not chunks:
            print(f"No text chunks generated for {file_path}. It might be empty or too short.")
            return

        print(f"Generated {len(chunks)} chunks from {file_path}.")

        # Prepare metadata and IDs for each chunk
        chunk_metadatas = []
        chunk_ids = []
        base_metadata = doc_metadata.copy() if doc_metadata else {}
        base_metadata['source'] = file_path # Add/overwrite source with the file path

        for i, chunk_text in enumerate(chunks):
            chunk_meta = base_metadata.copy()
            chunk_meta['chunk_num'] = i
            # Could add more metadata like page number if available from loader
            chunk_metadatas.append(chunk_meta)
            chunk_ids.append(f"{os.path.basename(file_path)}_chunk_{i}_{str(uuid.uuid4())[:8]}")

        self.add_text_batch(texts=chunks, metadatas=chunk_metadatas, ids=chunk_ids)


    def query(self, query_text: str, n_results: int = 3, query_filter: Optional[Dict] = None) -> Optional[Dict]:
        """
        Queries the vector store for documents similar to the query_text.

        Args:
            query_text (str): The text to query for.
            n_results (int): The number of results to return.
            query_filter (Optional[Dict]): A metadata filter for the query.
                                           Example: {"source": "my_specific_doc.txt"}

        Returns:
            Optional[Dict]: A dictionary containing 'ids', 'documents' (text chunks),
                            'metadatas', and 'distances' of the results, or None if error.
                            Example:
                            {
                                'ids': [['id1', 'id2']],
                                'documents': [['text_chunk1', 'text_chunk2']],
                                'metadatas': [[{'source': 'file.txt'}, {'source': 'file.txt'}]],
                                'distances': [[0.1, 0.2]]
                            }
        """
        if self.collection.count() == 0:
            print("Warning: Querying an empty collection.")
            # return None # Or return empty result structure

        print(f"Querying collection for: '{query_text[:50]}...' (n_results={n_results})")
        try:
            results = self.collection.query(
                query_texts=[query_text], # Chroma expects a list of query texts
                n_results=n_results,
                where=query_filter # Optional metadata filter
                # include=['documents', 'metadatas', 'distances'] # Default includes these
            )
            return results
        except Exception as e:
            print(f"Error querying Chroma collection: {e}")
            return None

    def get_collection_count(self) -> int:
        """Returns the number of items in the collection."""
        return self.collection.count()

    def delete_collection(self):
        """Deletes the entire collection from the persistent client."""
        print(f"Attempting to delete collection: {self.collection_name}")
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' deleted successfully.")
            # Recreate an empty collection for future use if desired, or set self.collection to None
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.sentence_transformer_ef
            )
        except Exception as e:
            print(f"Error deleting collection '{self.collection_name}': {e}")

    def reset_collection(self):
        """Deletes and recreates the collection, effectively clearing all data."""
        self.delete_collection()
        print(f"Collection '{self.collection_name}' has been reset (cleared and recreated).")

if __name__ == '__main__':
    print("--- Testing VectorStoreHandler ---")

    # Create a dummy file for testing
    dummy_file_path = "dummy_test_document.txt"
    with open(dummy_file_path, "w", encoding="utf-8") as f:
        f.write("This is the first sentence about apples. Apples are a healthy fruit.\n")
        f.write("The second paragraph discusses bananas. Bananas are yellow and sweet.\n")
        f.write("Finally, a note about oranges. Oranges contain vitamin C and are citrus fruits.")

    dummy_pdf_path = "dummy_test_document.pdf" # You'll need to create a dummy PDF manually or use a tool
    # For now, we'll just check if pypdf is installed
    try:
        import pypdf
        print("pypdf is installed. You can create a dummy_test_document.pdf for PDF testing.")
        # If you create dummy_test_document.pdf, uncomment the add_document call for it below
    except ImportError:
        print("pypdf not installed. PDF testing will be skipped.")


    # --- Test Initialization ---
    # Using a different directory for testing to not interfere with main data
    test_db_dir = "./test_chroma_db_data"
    vs_handler = VectorStoreHandler(collection_name="test_collection", persist_directory=test_db_dir)
    print(f"Initial collection count: {vs_handler.get_collection_count()}")

    # --- Test Adding Document ---
    vs_handler.add_document_from_file(dummy_file_path, doc_metadata={"category": "fruits_test"})
    print(f"Collection count after adding '{dummy_file_path}': {vs_handler.get_collection_count()}")

    # If you created a dummy PDF:
    # Create a simple PDF named dummy_test_document.pdf with some text for this to work.
    # if os.path.exists(dummy_pdf_path) and 'pypdf' in sys.modules:
    #     vs_handler.add_document_from_file(dummy_pdf_path, doc_metadata={"category": "pdf_fruits_test"})
    #     print(f"Collection count after adding '{dummy_pdf_path}': {vs_handler.get_collection_count()}")


    # --- Test Querying ---
    print("\n--- Querying for 'apples' ---")
    query_results_apples = vs_handler.query("information about apples", n_results=2)
    if query_results_apples and query_results_apples.get('documents'):
        for i, doc in enumerate(query_results_apples['documents'][0]):
            print(f"Result {i+1} (Distance: {query_results_apples['distances'][0][i]:.4f}):")
            print(f"  ID: {query_results_apples['ids'][0][i]}")
            print(f"  Metadata: {query_results_apples['metadatas'][0][i]}")
            print(f"  Text: {doc[:100]}...") # Print first 100 chars
    else:
        print("No results for 'apples' or error in query.")


    print("\n--- Querying for 'citrus fruits' with filter ---")
    query_results_citrus = vs_handler.query(
        "tell me about citrus",
        n_results=1,
        query_filter={"source": dummy_file_path} # Filter by source
    )
    if query_results_citrus and query_results_citrus.get('documents'):
        for i, doc in enumerate(query_results_citrus['documents'][0]):
            print(f"Result {i+1} (Distance: {query_results_citrus['distances'][0][i]:.4f}):")
            print(f"  ID: {query_results_citrus['ids'][0][i]}")
            print(f"  Metadata: {query_results_citrus['metadatas'][0][i]}")
            print(f"  Text: {doc[:100]}...")
    else:
        print("No results for 'citrus fruits' or error in query.")

    # --- Test Resetting Collection ---
    print("\n--- Resetting collection ---")
    vs_handler.reset_collection()
    print(f"Collection count after reset: {vs_handler.get_collection_count()}")

    # --- Clean up dummy file ---
    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)
    # If you created a dummy PDF, remove it too:
    # if os.path.exists(dummy_pdf_path):
    #     os.remove(dummy_pdf_path)

    # Clean up test database directory
    # import shutil
    # if os.path.exists(test_db_dir):
    #     shutil.rmtree(test_db_dir)
    # print(f"Cleaned up test database directory: {test_db_dir}")

    print("\n--- VectorStoreHandler Test Complete ---")