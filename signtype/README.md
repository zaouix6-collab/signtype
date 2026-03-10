# 🤟 SignType

**Type with sign language.** SignType uses your webcam to recognize ASL finger spelling and types letters directly into any focused application — browser, chat, terminal, anywhere.

Built for **deaf and hard-of-hearing users**. All feedback is visual. No audio.

## Quick Start

```bash
# 1. Install system dependencies (Arch Linux)
sudo pacman -S python python-gobject gtk4 gtk4-layer-shell ydotool

# 2. Set up ydotool
systemctl --user enable --now ydotool
sudo usermod -aG input $USER
# Log out and back in after this step

# 3. Install Python dependencies
cd signtype
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Train the model (first time only)
python training/preprocess_dataset.py    # Extract landmarks
python training/train_fingerspell.py     # Train classifier

# 5. Run
python main.py
```

## How It Works

1. **Camera** captures your hand via webcam
2. **MediaPipe** extracts 21 hand landmarks per frame
3. **ML Classifier** identifies which ASL letter you're signing
4. **Hold to confirm** — hold a sign steady for 400ms to type it
5. **ydotool** injects the keystroke into whatever app is focused

The system shows a floating overlay with:
- **Mode indicator** — always shows TYPING / COMMAND / IDLE
- **Confidence bar** — red → yellow → green as certainty increases
- **Detected letter** — large display of what the model sees
- **Typed buffer** — last 20 characters you've typed

## Supported Letters

The fingerspell model recognizes the **ASL alphabet** (A-Z), plus:
- **Space** — sign "SPACE"
- **Delete** — sign "DEL" to backspace
- **Nothing** — rest position (no typing)

> **Note:** J and Z require motion and are recognized through the dynamic gesture model (train separately with `python training/train_dynamic.py`).

## Settings

Open **http://127.0.0.1:7842** while the app is running:

- **Camera preview** — see what the model sees (MJPEG stream)
- **Confidence threshold** — minimum certainty to accept a letter (default: 70%)
- **Hold time** — how long to hold a sign before typing (default: 400ms)
- **Cooldown** — delay before allowing the same letter again (default: 600ms)
- **Test Typing** — sends "hello" to verify ydotool works

Settings are saved to `config.json` and hot-reloaded automatically.

## Troubleshooting

### Overlay doesn't float on top
The overlay uses `gtk4-layer-shell`. If it appears as a normal window:
```bash
# Verify the library is installed
find /usr -name "libgtk4-layer-shell*"
# Should show: /usr/lib/libgtk4-layer-shell.so
```
The app sets `LD_PRELOAD` automatically. If the overlay still doesn't float, check that `gtk4-layer-shell` is installed and the path matches.

### Overlay background isn't transparent / blurred
On **KDE Plasma Wayland**, the blur effect requires:
1. Go to **System Settings → Workspace Behavior → Desktop Effects**
2. Enable the **Blur** effect

Without blur enabled, the overlay renders as a solid dark rectangle (still functional, just not as pretty).

### Letters won't type into apps
```bash
# Check ydotool is running
systemctl --user status ydotool

# If not started:
systemctl --user enable --now ydotool

# Make sure you're in the input group
groups  # Should include "input"

# If not:
sudo usermod -aG input $USER
# Then log out and back in
```

### "wtype: Compositor does not support virtual keyboard"
This is expected on KDE Wayland. SignType automatically falls back to `ydotool` which works everywhere.

### Camera not detected
```bash
# List cameras
ls /dev/video*

# Try a different camera index in settings (http://127.0.0.1:7842)
# or change camera_index in config.json
```

## Project Structure

```
signtype/
├── main.py                    # Entry point — orchestrates everything
├── core/
│   ├── camera.py              # Threaded camera capture
│   ├── landmark_extractor.py  # MediaPipe hand landmarks
│   ├── fingerspell_classifier.py  # ASL letter recognition (MLP)
│   ├── text_injector.py       # Types into focused apps (ydotool)
│   ├── state_machine.py       # IDLE ↔ TYPING ↔ COMMAND modes
│   ├── gesture_classifier.py  # Static gesture recognition
│   ├── dynamic_classifier.py  # Dynamic gesture recognition (LSTM)
│   └── command_dispatcher.py  # Execute bound commands
├── feedback/
│   ├── buffer_overlay.py      # GTK4 floating overlay
│   ├── audio.py               # Visual notifications (no audio)
│   └── tray.py                # System tray icon
├── settings/
│   └── server.py              # Web settings UI (FastAPI)
├── training/
│   ├── preprocess_dataset.py  # Extract landmarks from dataset
│   ├── train_fingerspell.py   # Train ASL classifier
│   └── train_dynamic.py       # Train dynamic gestures
├── config.json                # User configuration
└── requirements.txt           # Python dependencies
```

## License

MIT
