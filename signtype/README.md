# SignType

An open-source background application that brings deaf and hard-of-hearing users closer to their first language — sign language — by enabling typing in any application using ASL finger spelling via webcam, and executing system shortcuts bound to custom gestures. No keyboard required.

**All feedback is visual.** This system is built for deaf users — there is no audio output of any kind.

## License

MIT License — see [LICENSE](LICENSE).

## Platform

- **Primary target**: Linux (Arch Linux, KDE Plasma)
- **Display server**: Wayland (tested on KDE Plasma). X11 support is partial.
- **All processing is local and offline.** No cloud APIs, no telemetry.

## Requirements

### System Dependencies (Arch Linux)

```bash
# Python 3.12 (required for MediaPipe — does NOT support 3.13+)
yay -S python312

# Wayland tools, GTK4 overlay, and system tray support
sudo pacman -S wtype ydotool gtk4-layer-shell python-gobject gtk4 gobject-introspection
```

> **Note on ydotool**: After installing, enable the daemon and add yourself to the `input` group:
> ```bash
> sudo systemctl enable --now ydotool
> sudo usermod -aG input $USER
> # Then log out and back in
> ```

### Python Setup

```bash
cd signtype
python3.12 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### ASL Alphabet Dataset

The finger spelling model trains on the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle (87,000 images, 29 classes).

Download via Kaggle CLI:
```bash
KAGGLE_API_TOKEN=your_token kaggle datasets download grassknoted/asl-alphabet \
  -p data/asl_alphabet_raw --unzip
```

Or download manually and extract into `data/asl_alphabet_raw/`.

### Train Models

```bash
source .venv/bin/activate
python training/preprocess_dataset.py   # Extract landmarks from images
python training/train_fingerspell.py    # Train MLP (target: ≥97% accuracy)
```

## Usage

```bash
source .venv/bin/activate
python main.py
```

On first launch, the system detects missing models and guides you through setup.

## Modes

- **Typing Mode**: Sign ASL letters → characters appear in floating buffer → confirm gesture injects text into active app
- **Command Mode**: Perform bound gestures → system shortcuts fire after confidence hold
- **Mode Switch**: Both open palms facing camera toggles between modes
- **Visual notifications**: All status changes, errors, and confirmations appear as toast messages on the overlay

## Settings

Open `http://localhost:7842` from the system tray menu to configure gesture bindings, confidence thresholds, and system settings.

## Architecture

```
signtype/
├── core/           # Camera, MediaPipe, classifiers, state machine, I/O
├── feedback/       # GTK4 overlay, visual notifications, system tray
├── training/       # Dataset preprocessing, MLP/LSTM training, gesture recorder
├── settings/       # FastAPI web UI for configuration
├── main.py         # 7-thread orchestration entry point
└── config.json     # Runtime configuration
```

Built for community retraining — swap the dataset and retrain for BSL, ArSL, or any other sign language with no code changes.

## Contributing

1. Replace images in `data/asl_alphabet_raw/` with your sign language dataset
2. Run `python training/preprocess_dataset.py` to extract landmarks
3. Run `python training/train_fingerspell.py` to train the classifier
4. The rest of the codebase works unchanged — all language-specific logic lives in the training data

## Permissions Note

Input injection uses `wtype` (text) and `ydotool` (hotkeys) which work without root on Wayland after `input` group setup. The optional X11 fallback (`pyautogui`) may need additional configuration.
