"""
RunPod Serverless worker for CosyVoice2 (text + reference voice -> speech).

Input  (matches build_payload() in app/services/cosyvoice_service.py):
    {
      "input": {
        "text": "Script to speak",
        "reference_audio": "<base64 of the sample voice>",
        "reference_audio_format": "wav" | "mp3" | "m4a",
        "prompt_text": ""          # optional transcript of the sample voice
      }
    }

Output (matches AUDIO_BASE64_KEYS in cosyvoice_service.py):
    { "audio_base64": "<base64 wav>", "sample_rate": 24000,
      "duration_seconds": 12.3, "format": "wav" }

If "prompt_text" is empty the worker uses CosyVoice cross-lingual cloning,
otherwise zero-shot cloning (usually a closer voice match).
"""

import base64
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback

sys.path.append("/app/CosyVoice")
sys.path.append("/app/CosyVoice/third_party/Matcha-TTS")

import runpod  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

MODEL_DIR = os.getenv("MODEL_DIR", "/app/CosyVoice/pretrained_models/CosyVoice2-0.5B")
MAX_PROMPT_SECONDS = 30  # CosyVoice works best with a reference under ~30 s
CHUNK_CHARS = 200  # CosyVoice cannot handle arbitrarily long text in one go

# The CosyVoice API changed between versions; support both.
try:
    from cosyvoice.cli.cosyvoice import AutoModel

    USE_AUTOMODEL = True
except ImportError:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav

    USE_AUTOMODEL = False

# Loaded once per worker, not once per request.
print("Loading CosyVoice model...", flush=True)
_t0 = time.time()
if USE_AUTOMODEL:
    model = AutoModel(model_dir=MODEL_DIR)
else:
    model = CosyVoice2(MODEL_DIR, load_jit=False, load_trt=False, fp16=False)
print(f"Model ready in {time.time() - _t0:.1f}s (sample rate {model.sample_rate})", flush=True)


def split_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split a long script into sentence-based chunks of roughly `limit` characters."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s*|\n+", text) if s and s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def prepare_prompt(b64_audio: str, fmt: str, workdir: str) -> str:
    """Decode the reference voice and convert it to 16 kHz mono wav."""
    if "base64," in b64_audio:  # tolerate data-URI prefixes
        b64_audio = b64_audio.split("base64,", 1)[1]
    fmt = re.sub(r"[^a-z0-9]", "", (fmt or "wav").lower()) or "wav"

    raw_path = os.path.join(workdir, f"reference_in.{fmt}")
    with open(raw_path, "wb") as f:
        f.write(base64.b64decode(b64_audio))

    wav_path = os.path.join(workdir, "reference_16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", raw_path, "-t", str(MAX_PROMPT_SECONDS),
         "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )
    return wav_path


def synthesize(text: str, prompt_wav: str, prompt_text: str) -> torch.Tensor:
    prompt = prompt_wav if USE_AUTOMODEL else load_wav(prompt_wav, 16000)
    pieces = []
    for chunk in split_text(text):
        if prompt_text:
            generator = model.inference_zero_shot(chunk, prompt_text, prompt, stream=False)
        else:
            generator = model.inference_cross_lingual(chunk, prompt, stream=False)
        for out in generator:
            pieces.append(out["tts_speech"])
    if not pieces:
        raise ValueError("CosyVoice produced no audio for this text")
    return torch.cat(pieces, dim=1)


def handler(job):
    data = job.get("input") or {}
    text = (data.get("text") or "").strip()
    reference = data.get("reference_audio")

    if not text:
        return {"error": "Missing 'text' in input"}
    if not reference:
        return {"error": "Missing 'reference_audio' in input"}

    try:
        with tempfile.TemporaryDirectory() as tmp:
            prompt_wav = prepare_prompt(reference, data.get("reference_audio_format", "wav"), tmp)
            speech = synthesize(text, prompt_wav, (data.get("prompt_text") or "").strip())

            out_path = os.path.join(tmp, "output.wav")
            torchaudio.save(out_path, speech.cpu(), model.sample_rate,
                            encoding="PCM_S", bits_per_sample=16)
            with open(out_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("ascii")

        return {
            "audio_base64": audio_b64,
            "sample_rate": model.sample_rate,
            "duration_seconds": round(speech.shape[1] / model.sample_rate, 2),
            "format": "wav",
        }
    except subprocess.CalledProcessError:
        return {"error": "Could not read the reference voice file. Try a clean .wav or .mp3."}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


runpod.serverless.start({"handler": handler})