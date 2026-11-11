
import os

# Set Whisper model cache directory to D drive
WHISPER_CACHE_DIR = r"D:\Whisper-Models"
os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = WHISPER_CACHE_DIR

import whisper
import requests
from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables from .env
load_dotenv()

# GROQ API accepts audio up to 30 seconds.
# We use 25-second pieces with a safety margin.
GROQ_PIECE_SECONDS = 25

# Whisper model configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small").strip()

# Fix incorrect model naming automatically
if WHISPER_MODEL == "whisper-large-v3":
    WHISPER_MODEL = "large-v3"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_STT_TRANSLATE_URL = (
    "https://api.groq.com/openai/v1/audio/transcriptions"
)

GROQ_MODEL = os.getenv(
    "GROQ_STT_MODEL",
    "saaras:v2.5"
)

_model = None


def load_model():
    """Load Whisper model only once."""

    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        print(f"Whisper cache directory: {WHISPER_CACHE_DIR}")

        _model = whisper.load_model(WHISPER_MODEL)

        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Transcribe an audio chunk using local Whisper."""

    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["text"]


def _send_to_groq(piece_path: str) -> str:
    """Send one audio piece to GROQ and return English transcript."""

    headers = {
        "api-subscription-key": GROQ_API_KEY
    }

    with open(piece_path, "rb") as f:

        files = {
            "file": (
                os.path.basename(piece_path),
                f,
                "audio/wav"
            )
        }

        data = {
            "model": GROQ_MODEL,
            "with_diarization": "false"
        }

        response = requests.post(
            GROQ_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120
        )

    if not response.ok:
        print(f"\nGROQ returned {response.status_code}")
        print(f"Response body: {response.text}\n")

        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_groq(chunk_path: str) -> str:
    """
    Split audio into 25-second pieces,
    send each piece to GROQ and combine transcripts.
    """

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set in environment / .env"
        )

    audio = AudioSegment.from_wav(chunk_path)

    piece_ms = GROQ_PIECE_SECONDS * 1000

    full_text = ""

    total_pieces = (
        len(audio) + piece_ms - 1
    ) // piece_ms

    for i, start in enumerate(
        range(0, len(audio), piece_ms)
    ):

        piece = audio[start:start + piece_ms]

        piece_path = f"{chunk_path}_sv_{i}.wav"

        piece.export(
            piece_path,
            format="wav"
        )

        try:
            print(
                f"  -> GROQ piece {i + 1}/{total_pieces} ..."
            )

            full_text += _send_to_groq(piece_path) + " "

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:
    """
    Route audio to Whisper or Groq.

    English  -> Whisper
    urdu -> GROQ
    """

    if language.lower() == "urdu":
        return transcribe_chunk_groq(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:
    """Transcribe all audio chunks and combine results."""

    full_transcript = ""

    engine = (
        "GROQ"
        if language.lower() == "urdu"
        else "Whisper"
    )

    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk {i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        full_transcript += text + " "

    print("Transcription complete.")

    return full_transcript.strip()
