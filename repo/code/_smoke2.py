import time

from ollama import Client

c = Client(host="http://127.0.0.1:11434", timeout=300)

t = time.time()
r = c.chat(
    model="qwen2.5:3b-instruct-q4_K_M",
    messages=[{"role": "user", "content": "say OK"}],
    options={"temperature": 0, "seed": 42},
)
print("warmup", round(time.time() - t, 2), "s", r["message"]["content"][:50])

big = "x" * 5000
prompt = 'JSON only: {"k":1}. ignore: ' + big
t = time.time()
try:
    r = c.chat(
        model="qwen2.5:3b-instruct-q4_K_M",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0, "seed": 42, "num_ctx": 8192},
    )
    print("big", round(time.time() - t, 2), "s", r["message"]["content"][:80])
except Exception as e:
    print("big ERR", round(time.time() - t, 2), "s", type(e).__name__, e)
