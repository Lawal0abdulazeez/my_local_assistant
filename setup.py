# setup.py
from setuptools import setup, find_packages

setup(
    name="my_local_assistant_framework", # Choose a unique name for PyPI
    version="0.1.0",
    author="Lawal Abdulazeez Faruq",
    author_email="Lawalabdulazeezfaruq@gmail.com",
    description="A framework for creating local, personal AI assistants.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Lawal0abdulazeez/my_local_assistant", # Replace
    packages=find_packages(include=['my_local_assistant', 'my_local_assistant.*']), # Finds your 'my_local_assistant' package
    install_requires=[
        "llama-cpp-python>=0.2.0",
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
        "pypdf>=3.0.0",
        "python-docx>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Or your chosen license
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",

    entry_points={
    'console_scripts': [
        'my-rag-assistant-cli=my_local_assistant.cli.rag_cli:main_cli_loop_entry',
    ],
},
)