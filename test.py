
from dotenv import load_dotenv

load_dotenv()  # MUST be before any core imports

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all


source = "https://www.youtube.com/watch?v=JM9jMNG-I7I"
language = "english"  # "english" or "urdu"


# Step 1: Process audio/video input
print("\nProcessing audio/video...")
chunks = process_input(source)


# Step 2: Transcribe audio
print("\nTranscribing audio...")
transcript = transcribe_all(chunks, language=language)

print("\n" + "=" * 60)
print("COMPLETE TRANSCRIPT")
print("=" * 60)
print(transcript)


# Step 3: Generate meeting title (No API call)
title = generate_title(transcript)


# Step 4: Generate summary (One Gemini API call)
print("\nGenerating summary...")
summary = summarize(transcript)

print("\n" + "=" * 60)
print(f"TITLE: {title}")
print("=" * 60)

print("\nSUMMARY")
print("-" * 60)
print(summary)


# Step 5: Extract action items, decisions and questions
# All three are generated in ONE Gemini API call
print("\nExtracting meeting information...")
extracted = extract_all(transcript)

action_items = extracted["action_items"]
decisions = extracted["decisions"]
questions = extracted["questions"]


# Step 6: Display results
print("\n" + "=" * 60)
print("ACTION ITEMS")
print("=" * 60)
print(action_items)


print("\n" + "=" * 60)
print("KEY TAKEAWAYS")
print("=" * 60)
print(decisions)


print("\n" + "=" * 60)
print("OPEN QUESTIONS")
print("=" * 60)
print(questions)

print("\n" + "=" * 60)
print("PROCESSING COMPLETED")
print("=" * 60)
