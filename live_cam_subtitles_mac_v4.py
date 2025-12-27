"""
ProjLaoLao — Live EN→ZH Subtitles for Video Calls (macOS)

A local, real-time English → Chinese subtitle overlay designed for video calls
(e.g. WeChat), optimised for phone screens and poor connections.

Features:
- Real-time speech recognition (Whisper, local)
- Real-time English → Chinese translation (local, MarianMT)
- HUGE, high-contrast Chinese subtitles (grandma-friendly)
- English transcript panel for verification
- Webcam preview
- Designed to be used with OBS Virtual Camera

Platform:
- macOS (Apple Silicon & Intel)
- Python 3.10+

Author: Mark Bo
License: MIT
"""

import sys
import time
import queue
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import sounddevice as sd
import webrtcvad
import cv2

from PyQt6 import QtCore, QtGui, QtWidgets
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer


# =============================
# CONFIG
# =============================
SAMPLE_RATE = 16000
CHANNELS = 1

# Force MacBook mic index (set None to use system default)
FORCE_INPUT_DEVICE_INDEX: Optional[int] = 1

VAD_AGGRESSIVENESS = 1
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

SILENCE_TO_FINALIZE = 0.45
MIN_UTTERANCE_SEC = 0.6

# Show last N finalized lines (Chinese + English panel)
MAX_ZH_LINES = 2
MAX_EN_LINES = 8

# Models
WHISPER_MODEL_SIZE = "base"   # "small" = better, slower
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"
MT_MODEL = "Helsinki-NLP/opus-mt-en-zh"

# Translation anti-repeat
MT_MAX_NEW_TOKENS = 80
MT_REPETITION_PENALTY = 1.15
MT_NO_REPEAT_NGRAM = 3

# UI sizing
SUB_FONT_SIZE = 64            # Chinese HUGE
SUB_LINE_H = 100
SUB_PAD = 32
SUB_BG_ALPHA = 220
SUB_Y_OFFSET_FROM_BOTTOM = 220

EN_PANEL_WIDTH_RATIO = 0.34   # right panel width as fraction of image
EN_FONT_SIZE = 22
EN_LINE_H = 30
EN_BG_ALPHA = 190

METER_W = 220
METER_H = 16


# =============================
# Messages
# =============================
@dataclass
class Update:
    finalized_zh: Optional[str] = None
    finalized_en: Optional[str] = None
    mic_rms: Optional[float] = None


# =============================
# Worker
# =============================
class Worker(QtCore.QObject):
    updated = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._stop = threading.Event()
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=240)

        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        self.whisper = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE
        )

        self.mt_tokenizer = MarianTokenizer.from_pretrained(MT_MODEL)
        self.mt_model = MarianMTModel.from_pretrained(MT_MODEL)

        self.in_speech = False
        self.utterance_audio: List[np.ndarray] = []
        self.last_voice_time = 0.0

        self.mic_level = 0.0

    def stop(self):
        self._stop.set()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            return
        mono = indata[:, 0].copy()

        rms = float(np.sqrt(np.mean(np.square(mono))) + 1e-12)
        rms_norm = min(1.0, rms * 12.0)
        self.mic_level = 0.85 * self.mic_level + 0.15 * rms_norm
        self.updated.emit(Update(mic_rms=self.mic_level))

        try:
            self.audio_q.put_nowait(mono)
        except queue.Full:
            try:
                _ = self.audio_q.get_nowait()
                self.audio_q.put_nowait(mono)
            except queue.Empty:
                pass

    def _is_speech(self, frame_i16: np.ndarray) -> bool:
        return self.vad.is_speech(frame_i16.tobytes(), SAMPLE_RATE)

    def _transcribe(self, audio_f32: np.ndarray) -> str:
        segments, _info = self.whisper.transcribe(
            audio_f32,
            language="en",
            vad_filter=False,
            beam_size=1,
            best_of=1,
            temperature=0.0
        )
        return "".join(seg.text for seg in segments).strip()

    def _translate(self, en: str) -> str:
        en = en.strip()
        if not en:
            return ""
        batch = self.mt_tokenizer([en], return_tensors="pt", padding=True, truncation=True)
        out = self.mt_model.generate(
            **batch,
            max_new_tokens=MT_MAX_NEW_TOKENS,
            num_beams=1,
            do_sample=False,
            repetition_penalty=MT_REPETITION_PENALTY,
            no_repeat_ngram_size=MT_NO_REPEAT_NGRAM,
        )
        zh = self.mt_tokenizer.decode(out[0], skip_special_tokens=True)
        return zh.strip()

    def run(self):
        while not self._stop.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            now = time.time()
            frame_i16 = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
            speech = self._is_speech(frame_i16)

            if speech:
                self.in_speech = True
                self.last_voice_time = now
                self.utterance_audio.append(chunk.copy())

            if self.in_speech and (now - self.last_voice_time) >= SILENCE_TO_FINALIZE:
                audio = np.concatenate(self.utterance_audio, axis=0)
                dur = audio.shape[0] / SAMPLE_RATE

                self.in_speech = False
                self.utterance_audio = []

                if dur >= MIN_UTTERANCE_SEC:
                    en_final = self._transcribe(audio.astype(np.float32))
                    zh_final = self._translate(en_final)
                    if zh_final:
                        self.updated.emit(Update(finalized_en=en_final, finalized_zh=zh_final))


# =============================
# UI
# =============================
class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grandma Mode v4 — Huge Chinese + English verification panel")
        self.resize(1200, 740)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color:black;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)

        self.zh_lines: List[str] = []
        self.en_lines: List[str] = []
        self.mic_level: float = 0.0

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0). Close other apps using the camera.")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def closeEvent(self, event):
        try:
            self.timer.stop()
            if self.cap:
                self.cap.release()
        finally:
            event.accept()

    def push_final(self, en: str, zh: str):
        zh = (zh or "").strip()
        en = (en or "").strip()
        if zh:
            self.zh_lines.append(zh)
            self.zh_lines = self.zh_lines[-MAX_ZH_LINES:]
        if en:
            self.en_lines.append(en)
            self.en_lines = self.en_lines[-MAX_EN_LINES:]

    def set_mic_level(self, level: float):
        self.mic_level = float(max(0.0, min(1.0, level)))

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)

        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # --- mic meter ---
        x0, y0 = 20, 20
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 140))
        painter.drawRoundedRect(QtCore.QRect(x0, y0, METER_W, METER_H), 8, 8)

        fill = int((METER_W - 4) * self.mic_level)
        painter.setBrush(QtGui.QColor(80, 200, 120, 220))
        painter.drawRoundedRect(QtCore.QRect(x0 + 2, y0 + 2, fill, METER_H - 4), 7, 7)

        painter.setPen(QtGui.QColor(255, 255, 255, 210))
        painter.setFont(QtGui.QFont("Menlo", 11))
        painter.drawText(x0, y0 + 28, "Mic level")

        # --- right English panel ---
        panel_w = int(w * EN_PANEL_WIDTH_RATIO)
        panel_rect = QtCore.QRect(w - panel_w - 20, 70, panel_w, h - 90)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, EN_BG_ALPHA))
        painter.drawRoundedRect(panel_rect, 18, 18)

        painter.setPen(QtGui.QColor(255, 255, 255, 230))
        painter.setFont(QtGui.QFont("Menlo", EN_FONT_SIZE, QtGui.QFont.Weight.Bold))
        painter.drawText(panel_rect.adjusted(16, 12, -16, -12),
                         int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop),
                         "English transcript (verify):")

        # transcript lines
        painter.setFont(QtGui.QFont("Menlo", EN_FONT_SIZE))
        y = panel_rect.top() + 52
        for line in self.en_lines[-MAX_EN_LINES:]:
            painter.setPen(QtGui.QColor(230, 230, 230, 230))
            painter.drawText(panel_rect.left() + 16, y, panel_rect.width() - 32, EN_LINE_H,
                             int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                             line)
            y += EN_LINE_H
            if y > panel_rect.bottom() - 20:
                break

        # --- huge Chinese subtitles (centered, higher) ---
        zh = self.zh_lines[-MAX_ZH_LINES:]
        if zh:
            painter.setFont(QtGui.QFont("PingFang SC", SUB_FONT_SIZE, QtGui.QFont.Weight.Bold))

            box_h = SUB_PAD * 2 + SUB_LINE_H * len(zh)
            y_box = h - box_h - SUB_Y_OFFSET_FROM_BOTTOM
            rect = QtCore.QRect(20, y_box, w - panel_w - 60, box_h)  # leave space for right panel

            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(0, 0, 0, SUB_BG_ALPHA))
            painter.drawRoundedRect(rect, 26, 26)

            y_txt = y_box + SUB_PAD
            for line in zh:
                painter.setPen(QtGui.QColor(255, 255, 255, 245))
                painter.drawText(
                    rect.left(),
                    y_txt,
                    rect.width(),
                    SUB_LINE_H,
                    int(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter),
                    line
                )
                y_txt += SUB_LINE_H

        painter.end()

        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        ))


# =============================
# Controller
# =============================
class Controller(QtCore.QObject):
    def __init__(self, window: Window):
        super().__init__()
        self.window = window

        self.worker = Worker()
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)

        self.worker.updated.connect(self.on_update)
        self.thread.started.connect(self.worker.run)

        self.stream = None

    def start(self):
        if FORCE_INPUT_DEVICE_INDEX is not None:
            sd.default.device = (FORCE_INPUT_DEVICE_INDEX, None)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=self.worker.audio_callback
        )
        self.stream.start()
        self.thread.start()

    def stop(self):
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        finally:
            self.worker.stop()
            self.thread.quit()
            self.thread.wait(1500)

    @QtCore.pyqtSlot(object)
    def on_update(self, upd: Update):
        if upd.mic_rms is not None:
            self.window.set_mic_level(upd.mic_rms)
        if upd.finalized_en is not None or upd.finalized_zh is not None:
            self.window.push_final(upd.finalized_en or "", upd.finalized_zh or "")


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()

    ctl = Controller(win)
    ctl.start()

    timer = QtCore.QTimer()
    timer.start(250)
    timer.timeout.connect(lambda: None)

    try:
        sys.exit(app.exec())
    finally:
        ctl.stop()


if __name__ == "__main__":
    main()