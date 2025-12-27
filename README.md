# ProjLaoLao — Live EN→ZH Subtitles for Video Calls (macOS)

**ProjLaoLao** is a local, real-time English → Chinese subtitle overlay designed
to help non-English speakers (e.g. grandparents) understand video calls.

It displays **large, high-contrast Chinese subtitles** on top of a webcam feed
and is intended to be used as a **virtual camera input via OBS** for apps like
**WeChat**.

No cloud APIs. No data leaves your computer.

---

## ✨ Features

- 🎙️ Local speech recognition (OpenAI Whisper via faster-whisper)
- 🌏 Local English → Chinese translation (MarianMT)
- 👵 “Grandma mode”: **very large subtitles**, readable on phone screens
- 🧠 English transcript panel for verification
- 🎥 Webcam preview with subtitle overlay
- 🔌 Works with WeChat, Zoom, Teams, FaceTime (via OBS Virtual Camera)

---

## 🖥️ Platform Support

- macOS only (tested on Apple Silicon)
- Python 3.10+
- Requires camera + microphone access

---

## 📦 Requirements

You will need:

- Python 3.10 or newer
- OBS Studio (for virtual camera)
- Homebrew (recommended)

### macOS dependencies
```bash
brew install ffmpeg portaudio


