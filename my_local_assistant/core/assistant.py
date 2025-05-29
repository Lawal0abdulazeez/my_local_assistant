# my_local_assistant/core/assistant.py

from .llm_handler import LLMHandler
from .vector_store import VectorStoreHandler
from typing import Optional, List, Dict

class RAGAssistant:
    """
    Orchestrates Retrieval Augmented Generation by combining
    a VectorStoreHandler for context retrieval and an LLMHandler
    for text generation.
    """
    def __init__(self,
                 llm_handler: LLMHandler,
                 vector_store_handler: VectorStoreHandler,
                 system_prompt_template: Optional[str] = None,
                 context_prompt_template: Optional[str] = None,
                 no_context_prompt_template: Optional[str] = None):
        """
        Initializes the RAGAssistant.

        Args:
            llm_handler (LLMHandler): An initialized LLMHandler instance.
            vector_store_handler (VectorStoreHandler): An initialized VectorStoreHandler instance.
            system_prompt_template (str, optional): A template for the system message.
                Should include placeholders like {persona}.
                Example: "You are {persona}, a helpful AI assistant. Use the provided context to answer the user's question."
            context_prompt_template (str, optional): A template for the prompt when context is found.
                Should include placeholders like {context_str}, {query_str}, and potentially {system_prompt}.
                Example for instruction models:
                "{system_prompt}\n\nCONTEXT:\n{context_str}\n\nUSER: {query_str}\nASSISTANT:"
            no_context_prompt_template (str, optional): A template for the prompt when no relevant context is found.
                Should include placeholders like {query_str} and potentially {system_prompt}.
                Example for instruction models:
                "{system_prompt}\n\nUSER: {query_str}\nASSISTANT:"
        """
        self.llm_handler = llm_handler
        self.vector_store_handler = vector_store_handler

        # Default prompt templates (suitable for many instruction-following models)
        self.system_prompt_template = system_prompt_template or \
            "You are a helpful AI assistant named {persona}. Answer the user's question based on the provided context. If the context doesn't contain the answer, say you don't know from the context."

        self.context_prompt_template = context_prompt_template or \
            "{system_prompt}\n\nBased on the following context:\n---CONTEXT START---\n{context_str}\n---CONTEXT END---\n\nAnswer this question: {query_str}\n\nASSISTANT:"

        self.no_context_prompt_template = no_context_prompt_template or \
            "{system_prompt}\n\nUSER: {query_str}\nASSISTANT:"

        self.persona = "Local Assistant" # Default persona

    def set_persona(self, persona_name: str):
        """Sets the persona for the assistant, used in system prompts."""
        self.persona = persona_name
        print(f"Assistant persona set to: {self.persona}")

    def _format_context(self, context_chunks: List[str]) -> str:
        """Formats a list of context chunks into a single string."""
        if not context_chunks:
            return "No relevant context found."
        return "\n\n".join(context_chunks)

    def generate_rag_response(self,
                              query: str,
                              n_context_results: int = 3,
                              context_filter: Optional[Dict] = None,
                              llm_temperature: Optional[float] = None,
                              llm_max_tokens: Optional[int] = None,
                              llm_stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generates a response using RAG.

        1. Retrieves relevant context from the vector store.
        2. Constructs a prompt using the context and query.
        3. Sends the prompt to the LLM to get a response.

        Args:
            query (str): The user's question or input.
            n_context_results (int): Number of context chunks to retrieve.
            context_filter (Optional[Dict]): Metadata filter for vector store query.
            llm_temperature (Optional[float]): Override LLM temperature.
            llm_max_tokens (Optional[int]): Override LLM max tokens.
            llm_stop_sequences (Optional[List[str]]): Override LLM stop sequences.

        Returns:
            str: The LLM's generated response.
        """
        print(f"\nProcessing query: '{query[:50]}...'")

        # 1. Retrieve context
        retrieved_context = self.vector_store_handler.query(
            query_text=query,
            n_results=n_context_results,
            query_filter=context_filter
        )

        context_str = ""
        final_prompt = ""
        system_message = self.system_prompt_template.format(persona=self.persona)

        if retrieved_context and retrieved_context.get('documents') and retrieved_context['documents'][0]:
            # We get a list of lists for documents, metadatas, etc. from Chroma query
            # For a single query_text, we are interested in the first element of these lists.
            context_chunks_texts = retrieved_context['documents'][0]
            context_str = self._format_context(context_chunks_texts)
            print(f"Retrieved {len(context_chunks_texts)} context chunk(s).")
            # print(f"Context: {context_str[:300]}...") # For debugging

            final_prompt = self.context_prompt_template.format(
                system_prompt=system_message,
                context_str=context_str,
                query_str=query
            )
        else:
            print("No relevant context found in vector store for this query.")
            final_prompt = self.no_context_prompt_template.format(
                system_prompt=system_message,
                query_str=query
            )

        # print(f"\n--- Final Prompt to LLM ---\n{final_prompt}\n--------------------------") # For debugging

        # 2. Generate response using LLM
        response = self.llm_handler.generate_response(
            prompt=final_prompt,
            temperature=llm_temperature, # Uses LLMHandler's default if None
            max_tokens=llm_max_tokens,   # Uses LLMHandler's default if None
            stop=llm_stop_sequences if llm_stop_sequences else ["\nUSER:", "\nASSISTANT:"] # Sensible defaults
        )

        return response.strip()

    def add_document_to_knowledge_base(self,
                                       file_path: str,
                                       chunk_size: int = 1000,
                                       chunk_overlap: int = 100,
                                       doc_metadata: Optional[Dict] = None):
        """
        Adds a document to the assistant's knowledge base (vector store).
        """
        print(f"Adding document to knowledge base: {file_path}")
        self.vector_store_handler.add_document_from_file(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            doc_metadata=doc_metadata
        )
        print(f"Document '{file_path}' processed. Knowledge base count: {self.vector_store_handler.get_collection_count()}")

    def clear_knowledge_base(self):
        """Clears all documents from the assistant's knowledge base."""
        print("Clearing knowledge base...")
        self.vector_store_handler.reset_collection()
        print(f"Knowledge base cleared. Item count: {self.vector_store_handler.get_collection_count()}")


if __name__ == '__main__':
    # This block is for basic direct testing of the RAGAssistant itself.
    # The example CLI script will be more comprehensive for user interaction.
    print("--- Basic RAGAssistant Direct Test ---")
    import os

    # --- Configuration (MUST BE SET BY USER) ---
    # Ensure you have a model downloaded in the 'models' directory
    MODEL_FILE_NAME = "mistral-7b-instruct-v0.2.Q2_K.gguf" # <--- !!! CHANGE THIS if different !!!
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", MODEL_FILE_NAME)
    VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "test_assistant_chroma_db") # Separate DB for this test
    TEST_DOCS_PATH = os.path.join(PROJECT_ROOT, "test_assistant_docs")

    if not os.path.exists(MODEL_PATH):
        print(f"FATAL: LLM Model not found at {MODEL_PATH}. Please download and place it there.")
        exit()

    # Create dummy docs for testing
    os.makedirs(TEST_DOCS_PATH, exist_ok=True)
    doc1_path = os.path.join(TEST_DOCS_PATH, "ai_basics.txt")
    with open(doc1_path, "w") as f:
        f.write("Artificial Intelligence (AI) is intelligence demonstrated by machines. "
                "Machine Learning (ML) is a subset of AI that allows systems to learn from data. "
                "Deep Learning (DL) is a subset of ML using neural networks.")

    doc2_path = os.path.join(TEST_DOCS_PATH, "project_zephyr.txt")
    with open(doc2_path, "w") as f:
        f.write("Project Zephyr is a new initiative focused on renewable energy. "
                "The main goal is to develop efficient solar panels. "
                "The project started in January 2023.")

    try:
        # 1. Initialize Handlers
        llm_h = LLMHandler(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=0) # CPU
        vs_h = VectorStoreHandler(collection_name="rag_test_collection", persist_directory=VECTOR_DB_PATH)

        # Clear any old data from previous test runs
        vs_h.reset_collection()

        # 2. Initialize RAGAssistant
        assistant = RAGAssistant(llm_handler=llm_h, vector_store_handler=vs_h)
        assistant.set_persona("TestBot")

        # 3. Add documents to knowledge base
        assistant.add_document_to_knowledge_base(doc1_path, doc_metadata={"topic": "AI"})
        assistant.add_document_to_knowledge_base(doc2_path, doc_metadata={"topic": "Energy"})

        # 4. Ask questions
        queries = [
            "What is Machine Learning?",
            "Tell me about Project Zephyr.",
            "When did Project Zephyr start?",
            "What is quantum computing?" # This topic is not in our docs
        ]

        for q in queries:
            print(f"\nYOU: {q}")
            response = assistant.generate_rag_response(q, n_context_results=2)
            print(f"ASSISTANT: {response}")

    except Exception as e:
        print(f"An error occurred during the RAGAssistant test: {e}")
    finally:
        # Clean up test docs and DB (optional, but good for repeated testing)
        import shutil
        if os.path.exists(TEST_DOCS_PATH):
            shutil.rmtree(TEST_DOCS_PATH)
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
        print("\nCleaned up test documents and database.")

    print("--- RAGAssistant Direct Test Complete ---")