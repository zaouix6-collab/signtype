# SignType

An open-source background application that enables deaf and hard-of-hearing users to type in any application using ASL finger spelling via webcam, and execute system shortcuts bound to custom gestures. No keyboard required.

## License

MIT License — see [LICENSE](LICENSE).

## Platform

- **Primary target**: Linux (Arch Linux, Ubuntu 22.04+)
- **Display server**: Wayland (KDE Plasma tested). X11 support is partial.
- **All processing is local and offline.** No cloud APIs, no telemetry.

## Requirements

### System Dependencies (Arch Linux)

```bash
# Python 3.12 (required for MediaPipe — does NOT support 3.13+)
yay -S python312

# Wayland tools and TTS engine
sudo pacman -S wtype ydotool espeak-ng gtk4-layer-shell
```

> **Note on ydotool**: After installing, you may need to set up a udev rule for non-root access:
> ```bash
> sudo usermod -aG input $USER
> # Then log out and back in
> ```

### Python Setup

```bash
cd signtype
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ASL Alphabet Dataset

The finger spelling model trains on the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from Kaggle (87,000 images, 29 classes).

> **License**: The dataset is available under a royalty-free license for non-commercial use. It is NOT bundled with this project — you must download it yourself.

1. Download from Kaggle
2. Extract into `data/asl_alphabet_raw/`
3. The first-run wizard will preprocess and train automatically

## Usage

```bash
source .venv/bin/activate
python main.py
```

On first launch, the system guides you through setup (dataset preprocessing, model training, motion gesture recording).

## Modes

- **Typing Mode**: Sign ASL letters → characters appear in floating buffer → confirm gesture injects text
- **Command Mode**: Perform bound gestures → system shortcuts fire after confidence hold
- **Mode Switch**: Both open palms facing camera toggles between modes

## Settings

Open `http://localhost:7842` from the system tray menu to configure gesture bindings, thresholds, and system settings.

## Architecture

Built for community retraining — swap the dataset and retrain for BSL, ArSL, or any other sign language with no code changes. See `training/` for the pipeline.

## Contributing

The architecture is documented for contributors to retrain models for other sign languages:

1. Replace images in `data/asl_alphabet_raw/` with your sign language dataset
2. Run `python training/preprocess_dataset.py` to extract landmarks
3. Run `python training/train_fingerspell.py` to train the classifier
4. The rest of the codebase works unchanged

## Permissions Note

The `keyboard` Python library (optional X11 fallback) requires root or specific udev rules on Linux. The primary input injection uses `wtype` (text) and `ydotool` (hotkeys) which work without root on Wayland after initial `input` group setup.
