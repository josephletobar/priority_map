from typing import Optional
import requests


class LlamaVlmClient:

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        """Initializes the client pointing to a designated host and port."""
        self.host = host
        self.port = port

    @property
    def endpoint(self) -> str:
        """Dynamically builds the endpoint URL from host and port variables."""
        return f"http://{self.host}:{self.port}/v1/chat/completions"

    def analyze(
        self,
        prompt: str,
        image_base64: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **extra_settings,
    ) -> str:
        """Sends a text prompt and an optional base64-encoded image to the local VLM server.

        Accepts arbitrary generation parameters via kwargs (e.g., top_p=0.95).
        """
        # Construct the basic content payload with the text prompt
        content_payload = [{"type": "text", "text": prompt}]

        if image_base64:
            content_payload.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        # Assemble the core OpenAI-compliant JSON structure
        payload = {
            "model": "local-vlm",
            "messages": [{"role": "user", "content": content_payload}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Dynamically append any extra sampling parameters passed by the user
        payload.update(extra_settings)

        try:
            response = requests.post(self.endpoint, json=payload, timeout=60)

            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
            else:
                return f"API Error ({response.status_code}): {response.text}"

        except requests.exceptions.ConnectionError:
            return f"Connection Error: Could not connect to server at {self.endpoint}. Is it running?"
        except Exception as e:
            return f"Unexpected Error processing request: {str(e)}"


# --- Updated Integrated Test Execution ---
if __name__ == "__main__":
    from llama_server_helper import LlamaServer

    # Paths directly from your local setup
    MODEL_PATH = "/Users/neal/Documents/llama/llama.cpp/models/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
    MMPROJ_PATH = "/Users/neal/Documents/llama/llama.cpp/models/Qwen3VL8B-mmproj-F32.gguf"

    # 1. Initialize backend server configuration
    llama_backend = LlamaServer(
        model_path=MODEL_PATH, mmproj_path=MMPROJ_PATH, port=8600
    )

    # 2. Extract network attributes explicitly to feed the independent client
    vlm_client = LlamaVlmClient(
        host=llama_backend.host, 
        port=llama_backend.port
    )

    try:
        if llama_backend.start():
            print(f"[*] Querying decoupled client endpoint: {vlm_client.endpoint}")
            
            response = vlm_client.analyze(
                prompt="Confirm network isolation is successful.",
                temperature=0.1,
                max_tokens=50
            )
            print(f"[+] Server Output:\n{response}\n")

    finally:
        llama_backend.stop()
