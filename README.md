# 🧭 W.I.T.N.E.S.S.
**Web-based Interrogation and Testimony via a Neural Engaged Speech System**

---

## Overview

**W.I.T.N.E.S.S.** is a fully offline, web-based simulation platform designed to emulate realistic police witness interviews. It combines a **local language model** (via [Ollama](https://ollama.ai/)) with open-source **speech technology** including **Piper TTS** and **Faster-Whisper STT**, producing authentic dialogue interactions between an interviewer and a virtual witness persona.

All AI logic, persona data, and audio processing occur **locally** — no external APIs or cloud services are required. The system features both **text-based** and **voice-based** interview modes with full speech-to-text and text-to-speech capabilities.

Developed on an M1 Mac using locally-run software. There is significant potential to enhance it further on other platforms using alternate software and internet-enabled services (such as cloud TTS/STT and LLM APIs).

### Feature Overview Video

https://github.com/user-attachments/assets/d4270fe3-2054-4447-9067-b6c0fcb0b79d

### Demonstration Video

https://vimeo.com/1177174879

---

## System Architecture

### Frontend
Static site served by **FastAPI** from `/frontend/`.  
All pages use a **black background with white text** (no light/dark mode).

#### Pages
| Page | Function |
|------|-----------|
| `index.html` | Home page with persona upload/validation before navigation to Create, Edit, or Interview modes. |
| `create-persona.html` | Create a new persona from template with inline field validation. |
| `edit-persona.html` | Edit an existing persona file with validation, export, and cancel safeguards. |
| `conduct-interview.html` | Main interaction page — conducts text or voice-based interviews with real-time audio. |
| `admin/index.html` | Admin dashboard providing access to system utilities (hidden from main navigation). |
| `admin/system-check.html` | Dependency version checker showing installed Python packages and system components. |
| `admin/voice-demos.html` | Voice preview tool for testing all available Piper TTS voices. |
| `admin/custom-dictionary.html` | Spell checker configuration for custom words and suspicious word warnings. |

#### Frontend Assets
```
frontend/
├── index.html
├── create-persona.html
├── edit-persona.html
├── conduct-interview.html
├── admin/
│   ├── index.html
│   ├── system-check.html
│   ├── voice-demos.html
│   └── custom-dictionary.html
├── styles.css
├── images/
└── temp_audio/  (auto-generated for TTS output)
```

---

### Sample Personas (Root Level)
Production-ready persona files for testing and demonstration:
```
sample_data/
├── Sally-Ann-Smith-1993-04-12.json
├── Rick-Sampson-1987-02-16.json
├── Mereana-Rangi-1990-06-14.json
├── Liam-O'Connor-1985-12-02.json
├── Priya-Patel-1993-09-07.json
├── Alex-Taylor-1978-04-19.json
└── Rob-McFlinty-2004-05-07.json
```
Each persona includes detailed background, facts with certainty/reason fields, and behavioral modifiers.

---

### Backend
Located in `/backend/`, powered by **FastAPI** (served via `uvicorn`).

#### Core Modules
| File | Purpose |
|------|----------|
| `api.py` | Main FastAPI app with all endpoints, text normalization functions (NZ spelling, TTS, Unicode safety), persona validation (4-stage check + spell checking), and route handlers. Runs on port `8010`. |
| `models/llm/llm_handler.py` | Hybrid deterministic + LLM response system. 30+ fact extraction functions, conversation memory, post-processing guardrails. Communicates with **local Ollama**. |
| `models/audio/audio_handler.py` | Speech-to-Text (Faster-Whisper) and Text-to-Speech (Piper) integrations with adaptive sample rate handling. |
| `persona_template.json` | JSON template defining required and optional persona fields. Used for validation and creation workflows. |
| `custom_dictionary.txt` | User-editable spell checker dictionary for persona names, NZ terms, and technical vocabulary. Hot-reloads on save. |
| `suspicious_words.txt` | List of valid words that should trigger warnings (e.g., common typos). Hot-reloads on save. |

#### Model Directory
```
backend/models/
├── audio/
│   ├── audio_handler.py
│   ├── stt/
│   │   └── whisper/  (Faster-Whisper cache)
│   └── tts/
│       └── piper-voices/  (ONNX voice models + configs)
├── common/
│   └── utils.py
└── llm/
    └── llm_handler.py
```

---

## Persona System

All personas are `.json` files following the `persona_template.json` structure.

### Required Key Fields (8 mandatory)
- `persona_type` (Witness/Suspect)
- `persona_voice_model` (ONNX filename)
- `persona_voice_speaker_id` (integer or null)
- `full_name`
- `date_of_birth`
- `home_address`
- `interview_instructions`
- `persona_prompt`

### Facts Structure
Each fact in `facts_to_provide` array contains:
- `fact` - The information the persona knows
- `certainty` - Confidence level (certain/pretty sure/unsure)
- `reason` - Why they have that certainty level

### Validation Logic
| Condition | Message |
|------------|----------|
| Wrong schema | 🔴 "Sorry, this is not a valid W.I.T.N.E.S.S. persona file." |
| Missing key fields | 🔴 "Sorry, this W.I.T.N.E.S.S. persona file is missing required fields and cannot be used." |
| All fields empty | 🔴 "Sorry, this W.I.T.N.E.S.S. persona file has no information in it and cannot be used." |
| Some fields empty | ⚠️ "This W.I.T.N.E.S.S. persona file is lacking detail. The interview may feel limited or unrealistic." |
| Spelling warnings | ⚠️ "Possible spelling errors detected" with list of flagged words. |
| Valid and complete | ✅ Displays name, DoB, address with Proceed/Cancel options. |

All validation messages appear inline under the relevant action buttons.
All content uses **New Zealand English** conventions (e.g., "111" emergency number, metric units).

### Spell Checking System
The system includes an integrated spell checker for persona files:
- **Custom Dictionary** (`backend/custom_dictionary.txt`) - Add persona names, NZ place names, Māori terms, and technical vocabulary
- **Suspicious Words** (`backend/suspicious_words.txt`) - Flag valid words that are commonly typos (e.g., "dunner" for "dinner")
- **Curly Quote Normalization** - Automatically handles both straight (`'`) and curly (`'`) apostrophes
- **Export Normalization** - All exported persona files use straight quotes to prevent compatibility issues
- **Admin Interface** - Manage dictionaries via `admin/custom-dictionary.html` with hot-reload on save

---

## AI Conversation System

The **LLM handler** is the reasoning engine of W.I.T.N.E.S.S.

### Functions
- Builds context-aware prompts using persona data.
- Maintains dialogue memory and contextual consistency.
- Enforces strict persona immersion (no “I am an AI” statements).
- Handles contradiction resolution, direction/distance reasoning, and object/scene coherence.
- Uses local **Ollama** instance for model execution.

### Environment Variable
```
OLLAMA_BASE_URL=http://localhost:11434
```

### Example Command
```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8010
```

---

## Audio System

### Features
- **Press-to-Talk** - Browser-based microphone recording with visual feedback
- **STT Pipeline** - WebM/Opus upload → WAV conversion → Faster-Whisper transcription
- **TTS Pipeline** - Piper synthesis with persona-specific voices and adaptive resampling
- **End-to-End Voice** - Complete speech-to-speech interview workflow
- **Text Normalization** - NZ English spelling, TTS number/date handling, Unicode safety

### Voice Models
Voice model files (`.onnx` + `.onnx.json`) are **NOT INCLUDED** in this repository due to their size. Place downloaded models in `backend/models/audio/tts/piper-voices/`. See the `README.md` and `piper_curated_audio.csv` in that folder for the list of voices used in v1.0 and the URLs from where you can download the voice files.

---

## Local Environment Setup

### Project Structure
```
witness/
├── backend/
├── frontend/
├── sample_data/
├── venv/
├── start.sh
└── requirements.txt
```

### Installation Steps

```bash
# 1. Create and activate venv (Python 3.10 recommended)
python3.10 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt
```

**3. Install system dependencies**

macOS (Homebrew):
```bash
brew install ollama ffmpeg espeak-ng
```

Linux — Ubuntu/Debian:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
sudo apt install ffmpeg espeak-ng libespeak-ng-dev
```

Linux — Fedora/RHEL:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
sudo dnf install ffmpeg espeak-ng espeak-ng-devel
```

> **Note for macOS users:** If Homebrew reports permission errors during install (e.g. for `/opt/homebrew/share/zsh`), fix them before continuing:
> ```bash
> sudo chown -R $(whoami) /opt/homebrew/share/man/man8 /opt/homebrew/share/zsh /opt/homebrew/share/zsh/site-functions
> brew upgrade
> ```

```bash
# 4. Pull Ollama model
ollama pull llama3.2:3b-instruct-q4_K_M

# 5. Set environment variables (optional — start.sh sets sensible defaults)
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.2:3b-instruct-q4_K_M
export PIPER_VOICES_DIR="$PWD/backend/models/audio/tts/piper-voices"
```

The espeak-ng library path must match your platform. `start.sh` defaults to the macOS Homebrew paths; Linux users should override before running:

macOS (Homebrew, Apple Silicon):
```bash
export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/lib/libespeak-ng.dylib
export ESPEAK_DATA_PATH=/opt/homebrew/share/espeak-ng-data
```

Linux (x86_64):
```bash
export PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1
export ESPEAK_DATA_PATH=/usr/lib/x86_64-linux-gnu/espeak-ng-data
```

Linux (ARM64 — e.g. Raspberry Pi):
```bash
export PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1
export ESPEAK_DATA_PATH=/usr/lib/aarch64-linux-gnu/espeak-ng-data
```

### Running the Application

**Recommended (Unified Startup):**
```bash
./start.sh
```
This script handles Ollama startup and launches the FastAPI server.

**Manual Startup:**
```bash
# Start Ollama separately, then:
uvicorn backend.api:app --host 0.0.0.0 --port 8010 --reload
```

### Access Points
- **Main Interface:** http://localhost:8010/
- **Admin Dashboard:** http://localhost:8010/admin/


No internet connection required during operation with the exception of the Admin Dashboard - System Check feature.

---

## Developer Guidelines

- **NZ English mandatory** - Use NZ spelling and context (metres, licence, 111, favour, colour)
- **No embedded personas** - Require user upload each session; personas stored in sessionStorage only
- **Scenario-agnostic design** - All fact extraction uses generic patterns; no hardcoded scenario references
- **Facts-first philosophy** - Attempt deterministic extraction before LLM calls
- **Text normalization required** - Apply `normalize_en_nz()`, `normalize_tts_nz()`, `normalize_unicode_safe()` to all outputs
- **Stateless backend** - Always send full message history; no server-side session state
- **Three-field facts** - All facts must include: fact, certainty, reason
- **CSS centralized** - All styles in `styles.css` unless inline specifically required
- **Frontend = static HTML/JS; Backend = FastAPI only** - Clear separation of concerns

---

## Licensing & Distribution

- This project is licensed under the **GNU General Public Licence v3.0 (GPL v3)**.
- Voice models are not distributed with this repository; only the metadata CSV (`piper_curated_audio.csv`) is included, with download URLs for each voice used.
- See the `LICENSE` file in the root of the repository for full licence terms.

---

## Key Technologies

- **Python:** 3.10.19 (3.10 recommended for stability)
- **LLM:** Ollama (llama3.2:3b-instruct-q4_K_M)
- **TTS:** Piper TTS 1.4.1 (ONNX voice models)
- **STT:** Faster-Whisper 1.2.1 (small.en with int8 quantization)
- **Inference Engine:** CTranslate2 4.6.3
- **Backend:** FastAPI 0.129.0 + Uvicorn 0.41.0 + Pydantic 2.12.5
- **ML Framework:** PyTorch 2.7.1
- **Frontend:** Vanilla HTML/CSS/JavaScript
- **Audio:** ffmpeg, espeak-ng (phonemizer)
- **Platform:** Developed on macOS M1 (Apple Silicon)

---

## Browser Compatibility

For full functionality, use a modern browser that supports the WebRTC and MediaRecorder APIs:
- **Google Chrome** (recommended)
- **Mozilla Firefox**

Voice mode (microphone recording) requires browser permission for `localhost:8010`. Safari has limited support for the audio formats used and is not recommended.

---

## Troubleshooting

### The system won't start
- Make sure Ollama is installed: `ollama --version`
- Make sure the correct model is downloaded: `ollama list`
- Check the terminal output from `start.sh` for error messages
- Visit the System Check page: `http://localhost:8010/admin/system-check.html`

### start.sh times out on the first run after a fresh install
Large ML packages (PyTorch, Transformers, Faster-Whisper) take longer to load on first launch than on subsequent runs. If `start.sh` reports **"Timed out waiting for FastAPI to become healthy"** with an empty log, this is the most likely cause.
- Wait a few seconds after `pip install` finishes, then run `./start.sh` again — the second run is significantly faster
- Check what FastAPI is actually doing: `cat /tmp/witness_fastapi.log`
- On macOS, ensure Homebrew finished upgrading cleanly (no permission errors) before running `./start.sh` — see the installation note above

### Voice mode isn't working
- Check that your browser has microphone permission for `localhost:8010`
- Make sure you are pressing and holding the microphone button while speaking
- Ensure voice model files (`.onnx` + `.onnx.json`) are present in `backend/models/audio/tts/piper-voices/`
- Use Chrome or Firefox — Safari has limited MediaRecorder support

### Persona won't validate
- Check the file is a `.json` file exported from W.I.T.N.E.S.S. (or matching the template structure)
- Ensure all 7 required fields are completed and non-empty
- Review any spelling warnings — these do not prevent validation but may indicate data entry errors

### Responses seem inconsistent or generic
- Review the `persona_prompt` — make sure it is specific and detailed
- Add more facts to `facts_to_provide`, particularly for details the persona should know precisely
- Check that certainty and reason fields are completed for each fact

### Transcript export fails
- Ensure the interview has at least one exchange before exporting
- Check that `python-docx` is installed: visible on the System Check page

---

## Credits

Developed by **Philip Roy**
Architectural and technical development support by **ChatGPT (OpenAI)** and **Claude (Anthropic)**
All design and testing localised for **New Zealand English** environment.

---
