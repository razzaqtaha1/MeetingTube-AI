
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    return Groq(
        api_key=api_key,
        timeout=60.0,
        max_retries=2
    )


def extract_all(transcript: str) -> dict:

    empty_result = {
        "action_items": "None found.",
        "decisions": "None found.",
        "questions": "None found."
    }

    # Check transcript
    print("\n========== EXTRACTION INPUT ==========")
    print("Transcript length:", len(transcript or ""))
    print("Transcript preview:", (transcript or "")[:500])
    print("======================================\n")

    if not transcript or not transcript.strip():
        print("WARNING: Empty transcript received!")
        return empty_result

    print("Extracting meeting information using Groq...")

    try:
        client = get_client()

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": """

You are an intelligent video and meeting analysis assistant.

Analyze the complete transcript and extract useful information.

1. ACTION ITEMS:
Extract practical tasks, recommendations, suggested
activities, learning steps, and useful actions from the
transcript.

For educational videos:
- Identify what the learner should practice.
- Identify concepts the learner should study further.
- Convert useful recommendations into actionable tasks.
- If the speaker explains a concept that can be practiced,
  suggest a relevant practice activity.

Generate 2-3 useful action items when the transcript
contains enough educational information.

Keep actions relevant to the transcript.
Do not invent speaker assignments or deadlines.

2. KEY TAKEAWAYS:
Extract the most important concepts, facts, explanations,
conclusions, lessons, and insights from the transcript.
For educational videos, this category is especially important.
Always capture the main concepts explained in the video.
Do not return "None found" when the transcript contains
useful information.

3. OPEN QUESTIONS:
Extract:
- Every question asked by the speaker.
- Questions that the video answers or explains.
- Questions raised indirectly by the topic.
- Important concepts that a learner may need to explore further.

For educational videos, generate 2-3 relevant learning
questions based on the transcript if no explicit questions
are present.

Do not invent facts. Questions must be related to
the actual transcript.

Return ONLY valid JSON:

{
    "action_items": "Bullet points here",
    "decisions": "Key takeaways here",
    "questions": "Bullet points here"
}

All values must be strings.
Do not use Markdown code fences.

"""
                },
                {
                    "role": "user",
                    "content": f"Analyze this transcript:\n\n{transcript}"
                }
            ],
            temperature=0.2,
            max_completion_tokens=1500
        )

        result = response.choices[0].message.content

        # Debug Groq response
        print("\n========== RAW GROQ EXTRACTION RESPONSE ==========")
        print(result)
        print("==================================================\n")

        if not result:
            raise ValueError("Empty response from Groq.")

        result = result.strip()

        # Remove accidental Markdown code fences
        if result.startswith("```"):
            result = result.replace("```json", "", 1)
            result = result.replace("```", "").strip()

        data = json.loads(result)

        extracted_data = {
            "action_items": str(data.get("action_items", "None found.")),
            "decisions": str(data.get("decisions", "None found.")),
            "questions": str(data.get("questions", "None found."))
        }

        print("Extraction completed successfully.")

        return extracted_data

    except Exception as e:
        print(f"Extraction failed: {e}")

        return {
            "action_items": "Extraction failed.",
            "decisions": "Extraction failed.",
            "questions": "Extraction failed."
        }
