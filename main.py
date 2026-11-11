
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_all
from core.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Video Assistant...")

    # Step 1: Process audio/video input
    chunks = process_input(source)

    # Step 2: Transcribe audio
    transcript = transcribe_all(chunks, language)

    print(f"Raw transcription (first 300 characters): {transcript[:300]}")

    # Step 3: Generate title
    title = generate_title(transcript)

    # Step 4: Generate summary
    summary = summarize(transcript)

    # Step 5: Extract action items, decisions and questions
    extracted = extract_all(transcript)

    action_item = extracted["action_items"]
    decisions = extracted["decisions"]
    questions = extracted["questions"]

    # Step 6: Build RAG chain
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":

    source = input("Enter YouTube URL or local file path: ").strip()

    language = input("Language (english/urdu): ").strip() or "english"

    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")

    print(f"\n📋 Summary:\n{result['summary']}")

    print(f"\n✅ Action Items:\n{result['action_items']}")

    print(f"\n📚 Key Takeaways:\n{result['key_decisions']}")

    print(f"\n❓ Open Questions:\n{result['open_questions']}")

    print("=" * 60)

    # Phase 2: Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")

    rag_chain = result["rag_chain"]

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)

        print(f"\n🤖 Assistant: {answer}\n")
