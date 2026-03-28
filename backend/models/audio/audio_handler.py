# W.I.T.N.E.S.S. - Web-based Interrogation and Testimony via a Neural Engaged Speech System
# Copyright (C) 2026 Philip Roy <https://www.bluengrey.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import csv
import os
import json
import wave
import tempfile
import traceback
import audioop
import gc
from fastapi import UploadFile
from pydub import AudioSegment
import re
from threading import Lock


# Matches hyphenated spelled-out sequences like "S-A-L-L-Y" so the TTS engine
# can add pauses between letters.
SPELLED_SEQ_RE = re.compile(r"\b([A-Za-z](?:\s*-\s*[A-Za-z]){1,})\b([\.,!\?])?")
# Matches space-separated uppercase sequences like "S A L L Y" for the same purpose.
SPACE_SPELLED_RE = re.compile(r"\b([A-Z](?:\s+[A-Z]){1,})\b([\.,!\?])?")

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small.en")
CT2_WHISPER_CACHE = os.getenv("CT2_WHISPER_CACHE")

_whisper_model = None

# ONNX sessions are cached to avoid the overhead of re-loading voice models on every request.
# Concurrent ONNX calls can cause segfaults on Apple Silicon, so both a per-session cache lock
# and a global synthesis lock are used.
_onnx_sessions = {}
_onnx_lock = Lock()
_synthesis_lock = Lock()

def _load_whisper():
    """Load the Faster-Whisper model, or return the cached instance if already loaded.

    int8 quantisation is used because the system runs fully on CPU — it halves memory
    usage and speeds up inference with negligible accuracy loss for interview-length audio.
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
            download_root=CT2_WHISPER_CACHE if CT2_WHISPER_CACHE else None,
        )
    except Exception as e:
        print(f"[audio_handler] ❌ Failed to load Whisper model")
        _whisper_model = None
    return _whisper_model


def _transcribe_with_whisper(wav_path: str) -> str:
    model = _load_whisper()
    if model is None:
        raise RuntimeError("Whisper model failed to load. Ensure 'faster-whisper' is installed and WHISPER_MODEL is valid.")
    segments, info = model.transcribe(
        wav_path,
        language="en",
        # VAD (Voice Activity Detection) trims silence, which greatly reduces hallucination
        # on short or quiet recordings.
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 200,
            "speech_pad_ms": 400,
            "threshold": 0.35,
        },
        beam_size=5,
        # Temperature fallback: if the top-beam decode has low confidence, Whisper retries
        # with progressively higher temperatures for diversity.
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        condition_on_previous_text=True,
        no_speech_threshold=0.3,
        # The initial_prompt biases Whisper's vocabulary toward NZ police interview terminology.
        # It is not shown to the user — it only influences the transcription.
        initial_prompt=(
            "This is a New Zealand police interview. The interviewer asks questions like: "
            "What is your full name? What is your date of birth? What is your address? "
            "Where do you work? What is your phone number? What is your email? "
            "Tell me what happened. What did you see? Can you describe the person? "
            "What were they wearing? What happened next? Did you call 111?"
        ),
    )
    return " ".join(seg.text.strip() for seg in segments).strip()

# Load Whisper at import time so the first transcription request does not incur startup delay.
try:
    _load_whisper()
except Exception:
    print("[audio_handler] ⚠️ Whisper preload failed; will attempt on first use.")

def transcribe_audio_file(file: UploadFile):
    temp_input_path = None
    temp_output_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_input:
            temp_input.write(file.file.read())
            temp_input_path = temp_input.name
        temp_output_path = temp_input_path.replace(".webm", ".wav")
        audio = AudioSegment.from_file(temp_input_path)
        # Whisper requires 16 kHz mono PCM WAV. Browser MediaRecorder produces WebM/Opus,
        # so we resample and convert here before passing to the model.
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(temp_output_path, format="wav", codec="pcm_s16le")

        result = _transcribe_with_whisper(temp_output_path)
        final_text = (result or "").strip()
        return final_text

    except Exception as e:
        print("Error during transcription:")
        traceback.print_exc()
        raise RuntimeError(f"Transcription failed: {e}")

    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)


import io

# --- TTS text normalisation (NZ phrasing + slowed spelling) ---
def _two_digit_words(n: int) -> str:
    ones = ["zero","one","two","three","four","five","six","seven","eight","nine"]
    teens = ["ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
    tens = ["","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
    if n < 10:
        return ones[n]
    if 10 <= n < 20:
        return teens[n-10]
    t, o = divmod(n, 10)
    return tens[t] + ("-" + ones[o] if o else "")

def _minutes_to_words(m: int) -> str:
    if m == 0:
        return ""
    if m < 10:
        ones = ["zero","one","two","three","four","five","six","seven","eight","nine"]
        return "oh " + ones[m]
    return _two_digit_words(m)

_DIGIT_WORDS = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

def _speak_digits(token: str) -> str:
    """Convert a string of digits into space-separated words: '123' -> 'one two three'."""
    return " ".join(_DIGIT_WORDS.get(ch, ch) for ch in token)

def _speak_phone_like(s: str) -> str:
    """Turn phone-like strings into digit-by-digit with light pauses at grouping boundaries.
    Examples:
      '123-456' -> 'one two three, four five six'
      '(09) 555 1234' -> 'zero nine, five five five, one two three four'
      '+64 21 123 4567' -> 'plus six four, two one, one two three, four five six seven'
    """
    # Keep only tokens that are digits or '+' treated as a word 'plus'; use separators as grouping commas
    groups = []
    buf = []
    def flush_buf():
        if buf:
            groups.append(_speak_digits("".join(buf)))
            buf.clear()
    for ch in s:
        if ch.isdigit():
            buf.append(ch)
        elif ch == '+':
            flush_buf()
            groups.append('plus')
        elif ch in ('-', ' ', '\u00A0', '\t', '(', ')'):
            # boundary: end current group if we have digits accumulated
            flush_buf()
        else:
            # unexpected symbol; end group and keep going without adding the symbol
            flush_buf()
    flush_buf()
    # Join groups with a brief pause (comma). Collapse any empty segments.
    groups = [g for g in groups if g]
    return ", ".join(groups) if groups else s

def _normalize_tts_nz(text: str) -> str:
    # 111 -> one one one
    text = re.sub(r"\b111\b", "one one one", text)
    # Years: 19xx -> nineteen xx ; 20xx -> twenty xx
    text = re.sub(r"\b19(\d{2})\b", lambda m: "nineteen " + _two_digit_words(int(m.group(1))), text)
    text = re.sub(r"\b20(\d{2})\b", lambda m: "twenty " + _two_digit_words(int(m.group(1))), text)

    # ---- Time normalisation so TTS says "five thirty pm" instead of "five point three zero pm" ----
    # Pattern 1: h[.:]mm(am|pm) with or without space (e.g., 5.30pm, 5:30 pm)
    def _time_with_minutes_repl(m: re.Match) -> str:
        h = int(m.group(1))
        mm = int(m.group(2))
        mer = m.group(3).lower()
        hour_words = _two_digit_words(h) if h != 0 else "twelve"
        mins_words = _minutes_to_words(mm)
        if mins_words:
            return f"{hour_words} {mins_words} {mer}"
        else:
            return f"{hour_words} {mer}"

    text = re.sub(r"\b(\d{1,2})[.:](\d{2})\s*(am|pm)\b", _time_with_minutes_repl, text, flags=re.IGNORECASE)

    # Pattern 2: h(am|pm) (e.g., 5pm -> five pm; 12am -> twelve am)
    def _hour_only_repl(m: re.Match) -> str:
        h = int(m.group(1))
        mer = m.group(2).lower()
        hour_words = _two_digit_words(h) if h != 0 else "twelve"
        return f"{hour_words} {mer}"

    text = re.sub(r"\b(\d{1,2})\s*(am|pm)\b", _hour_only_repl, text, flags=re.IGNORECASE)

    # Pattern 3 (optional): normalise colon/dot plus minutes without meridiem to a friendlier form for TTS
    # We avoid converting to am/pm to prevent factual changes; just turn 5.30 -> "five thirty"
    def _time_no_mer_repl(m: re.Match) -> str:
        h = int(m.group(1))
        mm = int(m.group(2))
        hour_words = _two_digit_words(h) if h != 0 else "twelve"
        mins_words = _minutes_to_words(mm)
        return f"{hour_words} {mins_words}" if mins_words else hour_words

    text = re.sub(r"\b(\d{1,2})[.:](\d{2})\b", _time_no_mer_repl, text)

    # Phone-like numbers: speak digits individually with light pauses between groups.
    # Pattern excludes times (no colon) and very short numbers already handled (e.g., 111 above).
    PHONE_LIKE = re.compile(r"(?<!\w)(\+?\d[\d\s\-()]{4,}\d)(?!\w)")
    def _phone_repl(m: re.Match) -> str:
        token = m.group(1)
        # Skip if it contains a ':' (likely a time) or if it's obviously a year-only token of 4 digits
        if ':' in token:
            return token
        digits_only = re.sub(r"[^0-9]", "", token)
        if len(digits_only) == 4 and (digits_only.startswith('19') or digits_only.startswith('20')):
            return token
        return _speak_phone_like(token)
    text = PHONE_LIKE.sub(_phone_repl, text)
    # Expand NZ abbreviation to New Zealand
    text = re.sub(r"\bNZ\b", "New Zealand", text)
    # Expand non-emergency number 105 to "ten five"
    text = re.sub(r"\b105\b", "ten five", text)
    return text

def _slow_spelled_sequences(text: str) -> str:
    """Turn S-A-L-L-Y into 'S. A. L. L. Y.' so TTS adds pauses.
    Also handles space-separated uppercase sequences: 'S A L L Y' → 'S. A. L. L. Y.'.
    """
    def repl(m: re.Match) -> str:
        token = m.group(1)
        trail = m.group(2) or ""
        letters = re.split(r"\s*-\s*", token)
        if all(len(l) == 1 and l.isalpha() for l in letters):
            spoken = ". ".join(ch.upper() for ch in letters)
            # If original token was immediately followed by punctuation, preserve it instead of adding another dot
            return spoken + (trail if trail else ".")
        return m.group(0)

    out = SPELLED_SEQ_RE.sub(repl, text)

    def repl_space(m: re.Match) -> str:
        token = m.group(1)
        trail = m.group(2) or ""
        letters = token.split()
        if all(len(l) == 1 and l.isalpha() for l in letters):
            spoken = ". ".join(l for l in letters)  # already uppercase by regex
            return spoken + (trail if trail else ".")
        return m.group(0)

    out = SPACE_SPELLED_RE.sub(repl_space, out)
    return out

def normalize_tts_text(text: str) -> str:
    """Apply all TTS normalisation to a string before passing it to Piper.

    Two passes: NZ number/time/phone pronunciation, then slowed letter-spelling.
    These are always applied; there are no toggle flags.
    """
    out = _normalize_tts_nz(text)
    out = _slow_spelled_sequences(out)
    return out

def _get_onnx_session(onnx_path: str):
    """Return a cached ONNX Runtime session for the given voice model, creating one if needed.

    Session creation is expensive (~0.5 s). The cache avoids that cost on repeated calls
    for the same voice. Thread safety is enforced with _onnx_lock.
    """
    with _onnx_lock:
        if onnx_path not in _onnx_sessions:
            import onnxruntime as ort
            # Single-threaded, sequential execution prevents the inter-thread race conditions
            # that produce segfaults on Apple Silicon when multiple ORT sessions are active.
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # Graph optimisations were found to cause instability on Apple Silicon (M-series).
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess_options.enable_cpu_mem_arena = False

            _onnx_sessions[onnx_path] = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"],
                sess_options=sess_options
            )
        return _onnx_sessions[onnx_path]

def _assert_voice_files(onnx_path: str, json_path: str):
    """Raise a clear error if Piper voice assets are missing."""
    if not os.path.isfile(onnx_path):
        raise RuntimeError(f"Piper ONNX not found at: {onnx_path}")
    if not os.path.isfile(json_path):
        raise RuntimeError(f"Piper JSON not found at: {json_path}")


def synthesize_speech(
    text: str,
    onnx_path: str,
    config_path: str,
    speaker_id: int | None = None,
) -> bytes:
    """
    Synthesize speech using Piper and return raw WAV bytes.
    - Respects per-voice inference settings (length_scale, noise_scale, noise_w) from the JSON config.
    - Preserves punctuation and enforces natural phrasing by splitting text into sentences and
      inserting a short configurable silence between them (default 0.40s).
    """
    try:
        # --- Resolve voice asset paths (accept filenames or absolute paths) ---
        # If config_path is empty/missing, derive it from the ONNX filename by convention
        if not config_path:
            base = onnx_path[:-5] if onnx_path.endswith(".onnx") else onnx_path
            config_path = base + ".onnx.json"
        voices_dir = os.getenv("PIPER_VOICES_DIR", VOICES_DIR_DEFAULT)
        # Normalise ONNX path
        if not os.path.isabs(onnx_path):
            onnx_path = os.path.join(voices_dir, onnx_path)
        # Normalise JSON config path
        if not os.path.isabs(config_path):
            config_path = os.path.join(voices_dir, config_path)

        # Fail fast with a clear message if assets are missing
        _assert_voice_files(onnx_path, config_path)
        import numpy as np  # used for silence padding
        from piper.voice import PiperVoice
        from piper.config import PiperConfig, SynthesisConfig

        # Piper voice JSON configs exist in two formats depending on their source:
        # - Flat: {"sample_rate": 22050, "espeak_voice": "en-gb", ...}
        # - Nested: {"audio": {"sample_rate": 22050}, "espeak": {"voice": "en-gb"}, ...}
        # Both are handled below by checking each location in preference order.
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)

        sample_rate = raw_config.get("sample_rate") or raw_config.get("audio", {}).get("sample_rate") or 22050
        espeak_voice = raw_config.get("espeak_voice") or raw_config.get("espeak", {}).get("voice")

        # Required for PiperConfig (present in official Piper voice JSONs)
        phoneme_id_map = raw_config.get("phoneme_id_map") or raw_config.get("phoneme_map") or {}
        phoneme_type = raw_config.get("phoneme_type", "espeak")
        try:
            num_symbols = int(raw_config.get("num_symbols"))
        except Exception:
            inferred = len(phoneme_id_map) if isinstance(phoneme_id_map, dict) else 0
            if inferred <= 0:
                raise RuntimeError("Voice JSON missing 'num_symbols' and could not infer from phoneme map")
            num_symbols = inferred
        num_speakers = int(raw_config.get("num_speakers", 1))

        cfg_kwargs = {
            "num_symbols": num_symbols,
            "num_speakers": num_speakers,
            "phoneme_id_map": phoneme_id_map,
            "phoneme_type": phoneme_type,
            "sample_rate": int(sample_rate),
        }
        # Optional extras from JSON
        if espeak_voice:
            cfg_kwargs["espeak_voice"] = espeak_voice
        if "speaker_id_map" in raw_config:
            cfg_kwargs["speaker_id_map"] = raw_config.get("speaker_id_map")

        config = PiperConfig(**cfg_kwargs)

        # Inference parameters control speaking speed (length_scale) and prosody variation.
        # Use per-voice values from the JSON when present; fall back to Piper defaults.
        # Note: older Piper voice JSONs use "noise_scale_w"; newer ones use "noise_w".
        infer = raw_config.get("inference", {})
        length_scale = float(infer.get("length_scale", 1.0))
        noise_scale = float(infer.get("noise_scale", 0.667))
        noise_w = float(infer.get("noise_w", infer.get("noise_scale_w", 0.8)))


        # A fresh ONNX session is created for every synthesis call rather than using the cache.
        # Session caching (_get_onnx_session) was found to cause memory corruption under load,
        # so it is intentionally bypassed here. The per-call overhead is acceptable given
        # interview cadence (one response every few seconds at most).
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.enable_cpu_mem_arena = False
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"], sess_options=sess_options)
        voice = PiperVoice(session, config)

        # ---- Prepare WAV buffer ----
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))

            # SynthesisConfig field name for noise-width changed between Piper releases.
            # Inspect the dataclass at runtime rather than hardcoding one version.
            sc_fields = getattr(SynthesisConfig, "__dataclass_fields__", {})
            field_names = set(sc_fields.keys()) if sc_fields else set()
            nsw_key = "noise_scale_w" if "noise_scale_w" in field_names else ("noise_w" if "noise_w" in field_names else None)

            syn_kwargs = {
                "noise_scale": float(noise_scale),
                "length_scale": float(length_scale),
            }
            if nsw_key:
                syn_kwargs[nsw_key] = float(noise_w)
            if speaker_id is not None and str(speaker_id).strip() != "" and "speaker_id" in field_names:
                syn_kwargs["speaker_id"] = int(speaker_id)

            tts_text = normalize_tts_text(text.strip())
            if not tts_text:
                tts_text = text.strip() or "Okay."

            # When the response includes a spelled-out name (e.g., "S-A-L-L-Y"), slightly slow
            # the overall pace so individual letters are distinct. 1.15× is subtle enough to
            # avoid sounding unnatural on surrounding words.
            try:
                text_has_spelling = bool(SPELLED_SEQ_RE.search(tts_text) or SPACE_SPELLED_RE.search(tts_text))
            except Exception:
                text_has_spelling = False
            if text_has_spelling:
                length_scale = float(length_scale) * 1.15

            syn_kwargs["length_scale"] = float(length_scale)
            syn_cfg = SynthesisConfig(**syn_kwargs)

            # The global lock serialises all synthesis calls — concurrent ONNX inference
            # causes segfaults even with separate session objects.
            with _synthesis_lock:
                try:
                    voice.synthesize_wav(tts_text, wav_file, syn_cfg)
                except Exception as e:
                    with _onnx_lock:
                        if onnx_path in _onnx_sessions:
                            del _onnx_sessions[onnx_path]
                    raise

        # Post-process the raw WAV: add optional leading silence and resample to a standard rate.
        try:
            wav_bytes = buffer.getvalue()

            leading_ms_env = os.getenv("LEADING_SILENCE_MS", "")
            try:
                LEADING_MS = int(leading_ms_env) if leading_ms_env else 60
            except Exception:
                LEADING_MS = 60

            target_rate_env = os.getenv("OUTPUT_SAMPLE_RATE", "")
            def _choose_target_rate(in_rate: int) -> int:
                # Resample to a rate that is a simple integer multiple of the source rate.
                # Simple ratios (×2, ×3) avoid aliasing artefacts that appear with irrational ratios.
                # 22,050 Hz → 44,100 Hz (×2)
                # 16,000 / 24,000 / 32,000 Hz → 48,000 Hz (×3 / ×2 / ×1.5)
                if in_rate == 22050:
                    return 44100
                if in_rate in (32000, 24000, 16000):
                    return 48000
                if in_rate in (44100, 48000):
                    return in_rate
                return 48000
            try:
                TARGET_RATE = int(target_rate_env) if target_rate_env else _choose_target_rate(framerate)
            except Exception:
                TARGET_RATE = _choose_target_rate(framerate)

            in_buf = io.BytesIO(wav_bytes)
            with wave.open(in_buf, "rb") as r:
                n_channels = r.getnchannels()
                sampwidth  = r.getsampwidth()
                framerate  = r.getframerate()
                n_frames   = r.getnframes()
                pcm        = r.readframes(n_frames)

            # audioop.ratecv requires 16-bit (2-byte) samples.
            if sampwidth != 2:
                try:
                    pcm = audioop.lin2lin(pcm, sampwidth, 2)
                    sampwidth = 2
                except Exception:
                    pass

            # Prepend leading silence if requested
            if LEADING_MS > 0 and sampwidth == 2 and n_channels in (1, 2):
                silence_frames = int((framerate * LEADING_MS) / 1000)
                pcm = (b"\x00\x00" * silence_frames * n_channels) + pcm

            if TARGET_RATE and TARGET_RATE != framerate and sampwidth == 2:
                try:
                    if n_channels == 2:
                        # audioop.ratecv operates on mono. Convert stereo to mono first.
                        pcm_mono = audioop.tomono(pcm, 2, 0.5, 0.5)
                        n_channels_out = 1
                    else:
                        pcm_mono = pcm
                        n_channels_out = n_channels
                    # weightA=1, weightB=1 applies a gentle first-order IIR low-pass filter
                    # during conversion, which reduces high-frequency aliasing artefacts.
                    pcm_rs, _ = audioop.ratecv(pcm_mono, 2, 1, framerate, TARGET_RATE, None, 1, 1)
                    pcm = pcm_rs
                    framerate = TARGET_RATE
                    n_channels = n_channels_out
                except Exception:
                    # If resample fails, keep original
                    pass

            out_buf = io.BytesIO()
            with wave.open(out_buf, "wb") as w:
                w.setnchannels(n_channels)
                w.setsampwidth(sampwidth)
                w.setframerate(framerate)
                w.writeframes(pcm)

            return out_buf.getvalue()
        except Exception:
            # If post-processing fails, return the raw Piper output rather than raising.
            # The audio will still be audible, just at the voice's native sample rate.
            return buffer.getvalue()
    except Exception as e:
        print("Error during Piper TTS synthesis:")
        traceback.print_exc()
        raise RuntimeError(f"Piper TTS generation failed: {e}")
    finally:
        # Explicitly delete the session and voice objects and run GC to release ONNX memory.
        # Without this, repeated synthesis calls accumulate GPU/CPU memory and eventually crash.
        try:
            del session
            del voice
            gc.collect()
        except:
            pass


VOICES_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "tts", "piper-voices")

def load_voice_list():
    """
    Load and return curated voice data from CSV as a list of dictionaries.
    Only includes relevant fields for UI display.
    """
    voices_dir = os.getenv("PIPER_VOICES_DIR", VOICES_DIR_DEFAULT)
    csv_path = os.path.join(voices_dir, "piper_curated_audio.csv")
    voice_list = []

    try:
        with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Strip the UTF-8 BOM that Excel adds to CSV files, which can appear as a
                # leading \ufeff on the first key and break dictionary lookups.
                row = {k.lstrip("\ufeff"): v for k, v in row.items()}
                voice_list.append({
                    "name": row.get("name", ""),
                    "gender": row.get("gender", ""),
                    "age-range": row.get("age-range", ""),
                    "accent": row.get("accent", ""),
                    "notes": row.get("notes", ""),
                    "onnx-filename": row.get("onnx-filename", "").strip(),
                    "json-filename": row.get("json-filename", "").strip(),
                    "speaker_id": row.get("speaker_id", "").strip(),
                    "source": row.get("source", "").strip()
                })
    except Exception as e:
        print("Failed to load curated voice list:")
        traceback.print_exc()
        raise RuntimeError(f"Could not read curated_audio.csv: {e}")

    return voice_list

def _warmup_piper():
    try:
        from piper.voice import PiperVoice
        from piper.config import PiperConfig, SynthesisConfig
        WARMUP_ONNX = "en_GB-alba-medium.onnx"
        voices_dir = os.getenv("PIPER_VOICES_DIR", VOICES_DIR_DEFAULT)
        onnx_path = os.path.join(voices_dir, WARMUP_ONNX)
        json_path = onnx_path + ".json"
        if not os.path.isfile(onnx_path):
            return
        with open(json_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)

        sample_rate = raw_config.get("sample_rate") or raw_config.get("audio", {}).get("sample_rate") or 22050
        espeak_voice = raw_config.get("espeak_voice") or raw_config.get("espeak", {}).get("voice")

        num_symbols = int(raw_config.get("num_symbols"))
        num_speakers = int(raw_config.get("num_speakers", 1))
        phoneme_id_map = raw_config.get("phoneme_id_map") or raw_config.get("phoneme_map") or {}
        phoneme_type = raw_config.get("phoneme_type", "espeak")

        cfg_kwargs = {
            "num_symbols": num_symbols,
            "num_speakers": num_speakers,
            "phoneme_id_map": phoneme_id_map,
            "phoneme_type": phoneme_type,
            "sample_rate": int(sample_rate),
        }
        if espeak_voice:
            cfg_kwargs["espeak_voice"] = espeak_voice
        if "speaker_id_map" in raw_config:
            cfg_kwargs["speaker_id_map"] = raw_config.get("speaker_id_map")

        config = PiperConfig(**cfg_kwargs)

        # Use cached session to prevent multiple session creation
        session = _get_onnx_session(onnx_path)
        piper_voice = PiperVoice(session, config)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))

            sc_fields = getattr(SynthesisConfig, "__dataclass_fields__", {})
            field_names = set(sc_fields.keys()) if sc_fields else set()
            nsw_key = "noise_scale_w" if "noise_scale_w" in field_names else ("noise_w" if "noise_w" in field_names else None)

            syn_kwargs = {
                "noise_scale": 0.667,
                "length_scale": 1.0,
            }
            if nsw_key:
                syn_kwargs[nsw_key] = 0.8

            syn_cfg = SynthesisConfig(**syn_kwargs)

            piper_voice.synthesize_wav(" ", wav_file, syn_cfg)  # silent warm-up (space only, not played or returned)

    except Exception:
        pass

# Optional Piper preload: runs one silent synthesis to load phonemiser data into memory.
# Disabled by default because it adds ~1 second to startup. Enable with PIPER_PRELOAD=1.
if os.getenv("PIPER_PRELOAD", "0") in ("1", "true", "TRUE", "yes", "YES"):
    try:
        _warmup_piper()
    except Exception:
        pass