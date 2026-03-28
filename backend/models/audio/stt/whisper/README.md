# Faster-Whisper STT Model

This is where W.I.T.N.E.S.S. stores the Faster-Whisper speech-to-text model. The folder is included in the repository as a placeholder — the model files themselves are not included because they are too large.

**You do not need to do anything manually.** When you start W.I.T.N.E.S.S. using `start.sh`, the environment is configured so that Faster-Whisper will automatically download the model into this folder the first time it is needed. An internet connection is required for that one-time download only. After that, the system runs fully offline.

## Model details

| Setting | Value |
|---|---|
| Model | `small.en` (English-only) |
| Quantisation | `int8` (CPU-optimised) |
| Library | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 1.2.1 |
| Source | Hugging Face — `Systran/faster-whisper-small.en` |

## Pre-downloading the model (optional)

If you would like to download the model before your first run (for example, to verify it works or to do so on a fast connection), run this from the root of the project:

```bash
source venv/bin/activate
python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8', download_root='backend/models/audio/stt/whisper')"
```
