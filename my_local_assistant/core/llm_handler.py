# my_local_assistant/core/llm_handler.py

from llama_cpp import Llama

import os

class LLMHandler:
    """
    Handles loading and interacting with a local GGUF-based LLM
    using the llama-cpp-python library.
    """
    def __init__(self,
                 model_path: str,
                 n_ctx: int = 2048,      # Context window size
                 n_gpu_layers: int = 0,  # Number of layers to offload to GPU (0 for CPU-only)
                 temperature: float = 0.7,
                 max_tokens: int = 512,
                 verbose: bool = False):
        """
        Initializes the LLMHandler.

        Args:
            model_path (str): Path to the GGUF model file.
            n_ctx (int): The context window size for the model.
            n_gpu_layers (int): Number of layers to offload to GPU. 0 for CPU.
                                 Set to a positive number if you have a compatible GPU
                                 and llama-cpp-python compiled with GPU support.
            temperature (float): Sampling temperature for generation.
            max_tokens (int): Maximum number of tokens to generate.
            verbose (bool): Whether to enable verbose output from llama.cpp.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        self.llm = None
        self._load_model()

    def _load_model(self):
        """Loads the LLM from the specified path."""
        print(f"Loading LLM from: {self.model_path}...")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose
            )
            print("LLM loaded successfully.")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            self.llm = None # Ensure llm is None if loading fails
            # You might want to re-raise the exception or handle it more gracefully
            raise

    def generate_response(self,
                          prompt: str,
                          temperature: float = None,
                          max_tokens: int = None,
                          stop: list[str] = None) -> str:
        """
        Generates a response from the LLM based on the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            temperature (float, optional): Override class temperature. Defaults to None.
            max_tokens (int, optional): Override class max_tokens. Defaults to None.
            stop (list[str], optional): List of strings to stop generation at.
                                         Helpful for instruction-following models.
                                         e.g., ["\nUSER:", "\nASSISTANT:"]

        Returns:
            str: The LLM's generated response.
        """
        if self.llm is None:
            print("LLM not loaded. Cannot generate response.")
            return "Error: LLM not loaded."

        current_temperature = temperature if temperature is not None else self.temperature
        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Default stop sequences can be useful, especially for instruction models
        # For a generic chat model, you might not need specific stop sequences
        # or just a newline.
        if stop is None:
            stop = ["\n"] # A simple default, adjust as needed for your model

        try:
            # print(f"\n--- Sending Prompt to LLM ---\n{prompt}\n---------------------------")
            output = self.llm(
                prompt,
                max_tokens=current_max_tokens,
                temperature=current_temperature,
                stop=stop, # Common stop sequences for instruction-tuned models
                echo=False # Do not echo the prompt in the output
            )
            # The output structure from llama-cpp-python's __call__ method:
            # {
            #   "id": "cmpl-...",
            #   "object": "text_completion",
            #   "created": 1677602606,
            #   "model": "/path/to/model",
            #   "choices": [
            #     {
            #       "text": "...",
            #       "index": 0,
            #       "logprobs": null,
            #       "finish_reason": "stop"
            #     }
            #   ],
            #   "usage": {
            #     "prompt_tokens": ...,
            #     "completion_tokens": ...,
            #     "total_tokens": ...
            #   }
            # }
            response_text = output["choices"][0]["text"].strip()
            # print(f"\n--- LLM Raw Output ---\n{output}\n---------------------------")
            # print(f"\n--- LLM Response ---\n{response_text}\n---------------------------")
            return response_text
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return f"Error generating response: {e}"

    def get_model_info(self) -> dict:
        """Returns basic information about the loaded model if available."""
        if self.llm:
            return {
                "model_path": self.model_path,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                # Add other relevant info if accessible from Llama object
            }
        return {"status": "LLM not loaded"}

if __name__ == '__main__':
    # This part is for testing the module directly, not for typical library usage.
    # You'd normally use the example script.
    print("Testing LLMHandler directly (not recommended for library use)...")
    # IMPORTANT: Replace with the actual path to YOUR GGUF model file
    # Ensure this model is in your project's `models/` directory or accessible
    MODEL_FILE_NAME = "mistral-7b-instruct-v0.2.Q2_K.gguf" # <--- CHANGE THIS
    # Construct the path relative to this script if needed, or use an absolute path.
    # For robust path handling, especially when running from different locations:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming 'models' is two levels up from 'my_local_assistant/core'
    # and then down into 'models/'
    # my_local_assistant/my_local_assistant/core/llm_handler.py
    # models/
    # So, ../../../models/
    project_root = os.path.abspath(os.path.join(current_script_dir, "..", "..", ".."))
    MODEL_PATH = os.path.join(project_root, "models", MODEL_FILE_NAME)


    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL: Model not found at {MODEL_PATH}")
        print("Please download a GGUF model, place it in the 'models' directory,")
        print(f"and update MODEL_FILE_NAME in {__file__} or ensure the path is correct.")
    else:
        try:
            handler = LLMHandler(model_path=MODEL_PATH, n_ctx=2048, verbose=True) # Enable verbose for loading
            if handler.llm: # Check if LLM loaded successfully
                test_prompt = "USER: What is the capital of France?\nASSISTANT:"
                # For instruct models, you often need to format the prompt correctly.
                # The prompt above is a common format.
                response = handler.generate_response(test_prompt, stop=["\nUSER:"])
                print("\nTest Prompt:", test_prompt)
                print("LLM Response:", response)

                another_prompt = "USER: Can you write a short poem about a cat?\nASSISTANT:"
                response_poem = handler.generate_response(another_prompt, stop=["\nUSER:"], max_tokens=100)
                print("\nTest Prompt:", another_prompt)
                print("LLM Response:", response_poem)
            else:
                print("LLM handler initialized, but LLM failed to load.")

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"An unexpected error occurred during direct testing: {e}")