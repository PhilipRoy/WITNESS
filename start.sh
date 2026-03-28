#!/bin/bash
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

# WITNESS unified startup script
# - Cleans caches, ensures Ollama is running, preloads model,
# - Activates venv, sets env vars, starts FastAPI, warms up, opens browser.
# - Designed for macOS (open) and Linux (xdg-open).

set -euo pipefail

# --- Resolve project root (this script lives in Source-Files) ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"  # Source-Files is the runtime root
cd "$PROJECT_DIR"          # ensure imports like 'backend' work

# --- Configurable knobs (override by exporting before running) ---
: "${WITNESS_PORT:=8010}"
: "${WITNESS_HOST:=0.0.0.0}"
: "${WITNESS_OPEN_BROWSER:=1}"  # set to 0 to disable auto-open

# LLM Configuration
: "${OLLAMA_MODEL:=llama3.2:3b-instruct-q4_K_M}"  # Preferred Ollama model
: "${OLLAMA_BASE_URL:=http://localhost:11434}"    # Base URL for local Ollama API

# Audio Configuration (optional - sensible defaults used if not set)
: "${WHISPER_MODEL:=small.en}"                    # Whisper STT model size
# : "${LEADING_SILENCE_MS:=60}"                   # TTS leading silence (ms)
# : "${OUTPUT_SAMPLE_RATE:=}"                     # Auto-detect if not set
# : "${PIPER_PRELOAD:=0}"                         # Set to 1 to preload voices

# --- Paths ---
VENV_DIR="$PROJECT_DIR/venv"
PIPER_VOICES_DIR="$PROJECT_DIR/backend/models/audio/tts/piper-voices"
CT2_WHISPER_CACHE="$PROJECT_DIR/backend/models/audio/stt/whisper"
FRONTEND_TEMP_AUDIO_DIR="$PROJECT_DIR/frontend/temp_audio"

FASTAPI_LOG="/tmp/witness_fastapi.log"
OLLAMA_LOG="/tmp/witness_ollama.log"
MODEL_WARMUP_LOG="/tmp/witness_model_warmup.log"

# espeak-ng paths for phonemizer/piper.
# Defaults below are correct for macOS with Homebrew.
# Linux users: override these before running start.sh, e.g.:
#   export PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1
#   export ESPEAK_DATA_PATH=/usr/lib/x86_64-linux-gnu/espeak-ng-data
: "${PHONEMIZER_ESPEAK_LIBRARY:=/opt/homebrew/lib/libespeak-ng.dylib}"
: "${ESPEAK_DATA_PATH:=/opt/homebrew/share/espeak-ng-data}"

# --- Helpers ---
log() { printf "[WITNESS] %s\n" "$*"; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Required command not found: $1" >&2; exit 1; }; }
wait_for_url() {
  local url="$1"; local tries="${2:-60}"; local delay="${3:-1}"
  for ((i=1;i<=tries;i++)); do
    if curl --silent --fail "$url" >/dev/null; then return 0; fi
    sleep "$delay"
  done
  return 1
}

# --- 1) Cleanup: caches & stray processes ---
log "Cleaning Python caches and stopping stray processes…"
find "$PROJECT_DIR" -name '*.pyc' -delete || true
find "$PROJECT_DIR" -type d -name '__pycache__' -exec rm -rf {} + || true
pkill -f ollama 2>/dev/null || true
pkill -f uvicorn 2>/dev/null || true
rm -f "$FRONTEND_TEMP_AUDIO_DIR"/response_*.wav 2>/dev/null || true
# --- 2) Ensure prerequisites (system tools) ---
require_cmd curl
require_cmd bash

# --- 3) Activate virtual environment ---
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Virtual environment not found at: $VENV_DIR" >&2
  echo "Create it first, e.g.: python3.10 -m venv '$VENV_DIR'" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
require_cmd python
require_cmd uvicorn

# --- 4) Start Ollama in background (if available) ---
if command -v ollama >/dev/null 2>&1; then
  log "Starting Ollama daemon… (logs: $OLLAMA_LOG)"
  nohup ollama serve >"$OLLAMA_LOG" 2>&1 &
  # Wait until API is responsive before preloading a model
  if wait_for_url "$OLLAMA_BASE_URL/api/tags" 30 1; then
    log "Preloading model: $OLLAMA_MODEL (logs: $MODEL_WARMUP_LOG)"
    nohup ollama run "$OLLAMA_MODEL" >"$MODEL_WARMUP_LOG" 2>&1 &
  else
    log "Warning: Ollama API not responding on $OLLAMA_BASE_URL; skipping preload."
  fi
else
  log "Ollama not found; skipping Ollama startup/preload."
fi

# --- 5) Export runtime environment variables ---
# LLM
export OLLAMA_BASE_URL
export OLLAMA_MODEL
# Audio (TTS/STT)
export PIPER_VOICES_DIR
export CT2_WHISPER_CACHE
export WHISPER_MODEL
# Phonemizer/espeak-ng (macOS Homebrew)
export PHONEMIZER_ESPEAK_LIBRARY
export ESPEAK_DATA_PATH

# --- 6) Start FastAPI (background) ---
log "Starting FastAPI on http://localhost:$WITNESS_PORT … (logs: $FASTAPI_LOG)"
nohup python -m uvicorn backend.api:app --host "$WITNESS_HOST" --port "$WITNESS_PORT" >"$FASTAPI_LOG" 2>&1 &
FASTAPI_PID=$!

# --- 7) Wait for /api/health ---
log "Waiting for API health…"
HEALTH_TRIES=0
until curl --silent --fail "http://localhost:$WITNESS_PORT/api/health" >/dev/null; do
  sleep 1
  HEALTH_TRIES=$((HEALTH_TRIES + 1))
  if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
    echo "FastAPI appears to have exited. Last 60 lines of log:" >&2
    tail -n 60 "$FASTAPI_LOG" >&2 || true
    exit 1
  fi
  if [[ "$HEALTH_TRIES" -ge 60 ]]; then
    echo "Timed out waiting for FastAPI to become healthy. Last 60 lines of log:" >&2
    tail -n 60 "$FASTAPI_LOG" >&2 || true
    exit 1
  fi
done

# --- 8) Trigger system warmup & wait for system-check ---
log "Triggering backend warmup…"
curl -s -X POST "http://localhost:$WITNESS_PORT/backend/system-warmup-trigger" >/dev/null || true

log "Waiting for system-check endpoint…"
until curl --silent --fail "http://localhost:$WITNESS_PORT/backend/system-check" >/dev/null; do
  sleep 1
  if ! kill -0 "$FASTAPI_PID" 2>/dev/null; then
    echo "FastAPI exited during warmup. Last 60 lines of log:" >&2
    tail -n 60 "$FASTAPI_LOG" >&2 || true
    exit 1
  fi
done

# --- 9) Optionally open browser ---
if [[ "$WITNESS_OPEN_BROWSER" == "1" ]]; then
  if command -v open >/dev/null 2>&1; then
    open "http://localhost:$WITNESS_PORT/"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:$WITNESS_PORT/"
  fi
fi

log "WITNESS is up. PID: $FASTAPI_PID"
wait "$FASTAPI_PID"