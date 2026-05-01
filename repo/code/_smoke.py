import json
import os
import time

from ollama import Client

c = Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))

t = time.time()
r = c.chat(
    model="qwen2.5:3b-instruct-q4_K_M",
    messages=[{"role": "user", "content": 'Reply ONLY with JSON: {"ok": true, "name": "qwen"}'}],
    format="json",
    options={"temperature": 0, "seed": 42},
)
print("chat latency:", round(time.time() - t, 2), "s")
print("chat content:", r["message"]["content"])
print("chat parsed:", json.loads(r["message"]["content"]))

t = time.time()
e = c.embeddings(model="nomic-embed-text", prompt="lost or stolen visa card")
print("embed dim:", len(e["embedding"]), "latency:", round(time.time() - t, 2), "s")
