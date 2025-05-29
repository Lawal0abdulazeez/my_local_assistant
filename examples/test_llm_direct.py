# examples/test_llm_direct.py

import os
import sys

# --- Add the parent directory of 'my_local_assistant' to the Python path ---
# This allows us to import from 'my_local_assistant'
# This is a common way to handle imports for example scripts
# when the library isn't installed yet.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..")) # Goes up one level to my_local_assistant/
# If my_local_assistant/ is the root containing the 'my_local_assistant' package directory
# then project_root itself should be in sys.path, or the directory *containing* project_root.
# Let's assume project_root is the directory containing the `my_local_assistant` package folder.
# e.g., your structure is:
# some_outer_folder/
# ├── my_local_assistant/    (this is project_root)
# │   ├── my_local_assistant/  (the actual package)
# │   │   └── core/
# │   │       └── llm_handler.py
# │   └── examples/
# │       └── test_llm_direct.py
# └── models/

sys.path.insert(0, project_root)
# --- End of path modification ---

try:
    from my_local_assistant.core.llm_handler import LLMHandler
except ImportError:
    print("Failed to import LLMHandler. Make sure your project structure is correct")
    print("and you are running this script from within the 'examples' directory,")
    print("or that 'my_local_assistant' is installed.")
    sys.exit(1)

# --- Configuration ---
# IMPORTANT: Replace with the actual path to YOUR GGUF model file
# Make sure the model file is in the 'models' directory at the project root.
MODEL_FILE_NAME = "mistral-7b-instruct-v0.2.Q2_K.gguf"  # <--- !!! CHANGE THIS !!!

# Construct the absolute path to the model
# Assuming 'models' directory is at the same level as the 'examples' and 'my_local_assistant' (package) directories
# i.e., project_root/models/your_model_name.gguf
MODEL_PATH = os.path.join(project_root, "models", MODEL_FILE_NAME)

def run_test():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'")
        print("Please download a GGUF model, place it in the 'models' directory,")
        print(f"and ensure MODEL_FILE_NAME in 'test_llm_direct.py' is correct.")
        return

    print(f"Attempting to load model from: {MODEL_PATH}")
    try:
        # Initialize the handler
        # For your 16GB RAM PC, n_gpu_layers=0 is crucial for CPU-only.
        # verbose=True can be helpful for debugging model loading.
        handler = LLMHandler(
            model_path=MODEL_PATH,
            n_ctx=2048,         # Or your model's preferred context size
            n_gpu_layers=0,     # Set to 0 for CPU
            verbose=False       # Set to True for more llama.cpp output
        )
    except Exception as e:
        print(f"Failed to initialize LLMHandler: {e}")
        return

    if not handler.llm:
        print("LLM failed to load within the handler. Exiting.")
        return

    print("\nLLM Handler Initialized. Type 'quit' or 'exit' to stop.")
    print("Example prompt formats for instruct models:")
    print("  USER: [Your question or instruction]\nASSISTANT:")
    print("  Simply type your question directly.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting...")
                break
            if not user_input.strip():
                continue

            # For many instruction-tuned models, you need a specific prompt format.
            # This is a common one. Adjust if your model expects something different.
            # If your model is a base model (not instruction-tuned), you might just pass `user_input`.
            # For Mistral-Instruct, this format works well.
            prompt_template = f"USER: {user_input}\nASSISTANT:"

            print("Assistant thinking...")
            response = handler.generate_response(
                prompt_template,
                temperature=0.7,
                max_tokens=300,
                stop=["\nUSER:"] # Stop generation if it tries to impersonate the user
            )
            print(f"Assistant: {response}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_test()
