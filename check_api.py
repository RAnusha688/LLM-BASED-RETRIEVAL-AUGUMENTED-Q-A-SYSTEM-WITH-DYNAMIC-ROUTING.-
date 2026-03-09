# check_api.py
import os
from openai import OpenAI

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("❌ OPENROUTER_API_KEY is NOT set")
    exit(1)

print("✅ API key found")

# OpenRouter client (THIS IS THE CRITICAL PART)
client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

try:
    response = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=5
    )
    print("✅ API key works!")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("❌ API key test failed:")
    print(e)
