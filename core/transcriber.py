
import os
from dotenv import load_dotenv
from groq import Groq
from pydub import AudioSegment

load_dotenv()

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")

# Maximum audio upload size
MAX_AUDIO_SIZE = 20 * 1024 * 1024

_client = None


def get_groq_client():
    """Initialize Groq client only once."""

    global _client

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Please check your .env file."
        )

    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)

    return _client


def _send_to_groq(audio_path: str, language: str = "english") -> str:
    """Send audio to Groq Cloud Whisper."""

    client = get_groq_client()

    language_code = {
        "english": "en",
        "urdu": "ur"
    }.get(language.lower())

    with open(audio_path, "rb") as audio_file:

        result = client.audio.transcriptions.create(
            file=audio_file,
            model=GROQ_MODEL,
            language=language_code,
            response_format="verbose_json",
            temperature=0
        )

    transcript = result.text.strip()

    print("\n========== GROQ RESPONSE DETAILS ==========")
    print("Audio Duration:", result.duration)
    print("Detected Language:", result.language)
    print("Transcript Length:", len(transcript))
    print("Transcript Ending:", transcript[-200:])
    print("===========================================\n")

    return transcript


def transcribe_chunk_groq(
    chunk_path: str,
    language: str = "english"
) -> str:
    """Transcribe audio directly or split oversized files."""

    if not os.path.exists(chunk_path):
        raise FileNotFoundError(
            f"Audio file not found: {chunk_path}"
        )

    file_size = os.path.getsize(chunk_path)

    # Direct upload for files below 20 MB
    if file_size <= MAX_AUDIO_SIZE:

        print("Sending audio directly to Groq...")

        return _send_to_groq(chunk_path, language)

    # Split large audio files
    print("Large audio detected. Splitting for Groq...")

    audio = AudioSegment.from_file(chunk_path)

    bytes_per_second = (
        audio.frame_rate * audio.frame_width * audio.channels
    )

    piece_ms = int(
        (MAX_AUDIO_SIZE / bytes_per_second) * 1000 * 0.75
    )

    full_text = ""

    total_pieces = (
        len(audio) + piece_ms - 1
    ) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):

        piece = audio[start:start + piece_ms]

        piece_path = f"{chunk_path}_groq_{i}.wav"

        try:
            piece.export(piece_path, format="wav")

            print(
                f"Groq transcription {i + 1}/{total_pieces}..."
            )

            text = _send_to_groq(piece_path, language)

            full_text += text + " "

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Compatibility wrapper using Groq instead of local Whisper."""

    return transcribe_chunk_groq(
        chunk_path,
        "english"
    )


def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:
    """Transcribe English and Urdu audio using Groq."""

    return transcribe_chunk_groq(
        chunk_path,
        language
    )


def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:
    """Transcribe all audio chunks and combine results."""

    full_transcript = ""

    print(
        f"Using Groq Cloud Whisper for {language} transcription."
    )

    for i, chunk in enumerate(chunks):

        print(
            f"\nTranscribing chunk {i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        full_transcript += text + " "

    print("\nTranscription complete.")

    final_transcript = full_transcript.strip()

    print("Final Transcript Length:", len(final_transcript))

    return final_transcript
