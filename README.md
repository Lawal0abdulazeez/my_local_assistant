# My Local Assistant Framework

A Python framework for building local, personal AI assistants with Retrieval Augmented Generation (RAG).

**Current Version: 0.1.0 (Early Alpha)**

This framework provides core components for:
- Interacting with local LLMs (via `llama-cpp-python`).
- Creating and querying vector stores for RAG (via `sentence-transformers` and `ChromaDB`).
- Orchestrating RAG pipelines.

## Installation (from TestPyPI for now)

```bash
pip install --index-url https://test.pypi.org/simple/ my-local-assistant-framework