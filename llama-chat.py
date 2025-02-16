import os
import sys
import httpx
from llama_stack_client import LlamaStackClient
from sty import fg

# These environment variables should be defined before running the script
# os.environ['LLAMA_STACK_PORT'] = "5001"
# os.environ['INFERENCE_MODEL'] = "meta-llama/Llama-3.2-1B-Instruct"

# This chat will work with the LLaMA Stack running locally on port 5001.
# You should use the --gpus all flag when running the LLaMA Stack to enable
# GPU support.
# The INFERENCE_MODEL environment variable should be set to the name of the
# model you want to use which should be an instruct model otherwise it will
# take a long time to respond and will probably timeout.

# See the start-llama.ps1 for an example of how to start the LLaMA Stack with

# Create the HTTP client
def create_http_client():
    return LlamaStackClient(
        base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}",
        timeout=httpx.Timeout(60.0)  # Set timeout to 60 seconds
    )

# Initialize the client
client = create_http_client()

# Function to get chat completion from LlamaStack
def get_llama_response(client, messages):
    response = client.inference.chat_completion(
        model_id=os.environ["INFERENCE_MODEL"],
        messages=messages,
    )
    return response.completion_message.content

# Main function to run the chatbot
def main():
    print("Welcome to the LLaMA Chatbot!")
    print("Type 'exit' to end the conversation.")
    
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        messages.append({"role": "user", "content": user_input})
        llama_response = get_llama_response(client, messages)
        messages.append({"role": "assistant", "content": llama_response, "stop_reason": "end_of_turn"})
        
        print(fg.yellow + f"LLaMA: {llama_response}" + fg.rs)

if __name__ == "__main__":
    main()