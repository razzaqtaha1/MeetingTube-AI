
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY missing in .env")

client = Groq(api_key=api_key)

model = "openai/gpt-oss-20b"

# Check model availability
models = client.models.list()
available_models = [m.id for m in models.data]

if model not in available_models:
    print("Model unavailable.")
    print("Available models:", available_models)
else:
    print("Testing Groq text generation...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful meeting assistant."
            },
            {
                "role": "user",
                "content": "Summarize this in 2 lines: "
                           "Software engineers design, develop, test "
                           "and maintain software applications."
            }
        ],
        temperature=0.2,
        max_completion_tokens=300
    )

    print("\nGROQ RESPONSE:")
    print(response.choices[0].message.content)
