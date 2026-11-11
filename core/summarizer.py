
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    return Groq(
        api_key=api_key,
        timeout=30.0,
        max_retries=1
    )


def generate_title(transcript: str) -> str:
    """Generate title without an API call."""

    if not transcript or not transcript.strip():
        return "Untitled Meeting"

    words = transcript.strip().split()

    return " ".join(words[:8]) + (
        "..." if len(words) > 8 else ""
    )


def summarize(transcript: str) -> str:
    """Generate meeting summary using Groq."""

    if not transcript or not transcript.strip():
        return "No transcript available."

    print("Generating summary using Groq...")

    try:
        client = get_client()

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert meeting summarizer. "
                        "Summarize the complete transcript using "
                        "clear headings and bullet points. "
                        "Include important topics, details and "
                        "conclusions. Do not invent information. "
                        "Keep the summary within 500 words."
                    )
                },
                {
                    "role": "user",
                    "content": f"Meeting transcript:\n{transcript}"
                }
            ],
            temperature=0.2,
            max_completion_tokens=1024
        )

        result = response.choices[0].message.content

        if not result:
            return "Empty summary received from Groq."

        return " ".join(result.split()[:500])

    except Exception as e:
        print(f"Summary generation failed: {e}")
        return "Summary could not be generated. Please try again."
