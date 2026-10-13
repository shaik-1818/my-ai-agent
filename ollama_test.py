import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:8b",
        "prompt": "What is 2 + 2? Answer with only the number.",
        "stream": False
    },
    timeout=120
)

response.raise_for_status()

data = response.json()

print("AI response:")
print(data["response"])