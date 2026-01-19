import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-ae06b64c707d4e0d902a57b0b5b49956",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who are you?"},
    ]
)
print(completion.model_dump_json())