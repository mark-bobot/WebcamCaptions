ProjLaoLao — How to Run (macOS)

This project shows real-time English → Chinese subtitles on a webcam feed.
It is designed to be used as a virtual camera input for WeChat (via OBS).

No cloud APIs. Everything runs locally.


=
SYSTEM REQUIREMENTS
=

- macOS
- Python 3.10 or newer
- Camera + Microphone
- Internet (first run only, to download AI models)

Recommended:
- Apple Silicon Mac (M1 / M2 / M3)

=
STEP 1 — INSTALL SYSTEM DEPENDENCIES
=

If you do not have Homebrew:
https://brew.sh

Then run:

brew install ffmpeg portaudio


=
STEP 2 — CREATE A VIRTUAL ENVIRONMENT
=

Open Terminal and go to the project folder:

cd /Applications/ProjLaoLao
# or wherever you saved the project

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate


=
STEP 3 — INSTALL PYTHON PACKAGES
=

Run:

pip install --upgrade pip

pip install numpy sounddevice webrtcvad opencv-python PyQt6 \
            faster-whisper transformers sentencepiece torch sacremoses

=
STEP 4 — RUN THE SCRIPT
=

Run:

python live_cam_subtitles_mac_v4.py

FIRST RUN NOTES:
- The first run will download speech + translation models (hundreds of MB)
- This only happens once
- The app may look like it is “doing nothing” for 30–90 seconds — this is normal
<img width="1288" height="946" alt="Screenshot 2025-12-27 at 14 29 22 (2)" src="https://github.com/user-attachments/assets/d292befc-3fe5-4cfd-a646-d28d7a900199" />


=
STEP 5 — macOS PERMISSIONS (IMPORTANT)
=

When prompted, ALLOW:
- Camera access
- Microphone access

If the camera is black or subtitles do not appear:

System Settings → Privacy & Security
- Camera → enable Terminal
- Microphone → enable Terminal

Then:
- Quit Terminal completely
- Reopen Terminal
- Run the script again


=
STEP 6 — USE IT IN WECHAT (VIRTUAL CAMERA)
=

This script does NOT create a camera device by itself.
You must use OBS Virtual Camera.

1) Install OBS:
https://obsproject.com

2) Open OBS
Click “Start Virtual Camera”

3) Add the subtitle window:
Sources → + → Window Capture
Select the ProjLaoLao window
Resize to fit canvas
Mirror the video so the orientation is correct in the call
<img width="1084" height="755" alt="Screenshot 2025-12-27 at 14 27 02" src="https://github.com/user-attachments/assets/15c30179-4932-49fa-b51c-bf459873de18" />


4) Open WeChat video call
Select camera:
OBS Virtual Camera

Your call partner will now see:
- Your webcam
- Huge Chinese subtitles

<img width="293" height="503" alt="Screenshot 2025-12-27 at 14 45 16" src="https://github.com/user-attachments/assets/6040ce28-ea11-453e-b6c0-3c12ffad0196" />


=
TROUBLESHOOTING
=

No subtitles:
- Speak clearly for ~4 seconds
- Pause for ~0.5 seconds (this finalizes a line)

Wrong microphone:
- Make sure MacBook microphone is selected
- AirPods can steal the input

Camera busy error:
- Close FaceTime / Zoom / Photo Booth
- Only one app can use the camera at a time


=
CUSTOMIZATION (OPTIONAL)
=

Open the Python file and edit:

SUB_FONT_SIZE = 64     # bigger number = bigger subtitles
SUB_Y_OFFSET_FROM_BOTTOM = 220
MAX_ZH_LINES = 2

Save the file and re-run the script.


=
STOPPING THE APP
=

Click the Terminal window and press:

Ctrl + C

This safely stops the program.

=
END
=
