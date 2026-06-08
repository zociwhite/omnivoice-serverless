# OmniVoice Serverless — RunPod STT & TTS API

Serverless deployment of OmniVoice-Studio on RunPod.
- STT: WhisperX large-v3 + pyannote speaker diarization
- TTS: OmniVoice (646 languages, voice cloning/design)

## Environment Variables
- `TASK_TYPE`: `stt` or `tts`
- `HF_TOKEN`: HuggingFace token (for pyannote)
