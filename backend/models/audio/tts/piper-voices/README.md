# Piper Voice Models

This folder is where Piper TTS voice model files must be placed before running W.I.T.N.E.S.S.

Each voice requires two files:
- `<voice-name>.onnx` — the model weights
- `<voice-name>.onnx.json` — the voice configuration

## Voices used in W.I.T.N.E.S.S. v1.0

The file `piper_curated_audio.csv` lists all voices included in the first release of W.I.T.N.E.S.S., along with download URLs for each. Download the `.onnx` and `.onnx.json` files for any voice you want to use and place them in this folder.

## Compatibility

Voice models were tested with **Piper TTS 1.4.1**. Models are in ONNX format and run entirely on CPU — no GPU required.

Most voices are sourced from the [Rhasspy Piper voices repository](https://huggingface.co/rhasspy/piper-voices/tree/main/en) on Hugging Face.

## Adding your own voices

Any Piper-compatible `.onnx` / `.onnx.json` voice pair can be added to this folder. Once placed here, the voice will appear automatically in the W.I.T.N.E.S.S. voice selector (Admin → Voice Demos). Add a corresponding row to `piper_curated_audio.csv` to include it in the curated list.
