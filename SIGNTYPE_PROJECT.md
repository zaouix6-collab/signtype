# SignType — Project Brief for Coding Agent

## What You Are Building

A background application for deaf and hard-of-hearing users that enables two things: typing in any application using ASL finger spelling captured via webcam, and executing custom system shortcuts bound to user-defined gestures. There is no keyboard required for either function once the system is set up.

The application has two runtime modes — Typing Mode and Command Mode — that the user switches between explicitly. A local web interface running on localhost handles all one-time configuration and gesture binding setup. The webcam pipeline, finger spelling recognition, and command dispatch all run silently in the background.

This is an open source project targeting Linux first, with the architecture kept clean enough for the community to retrain models for other sign languages beyond ASL.

---

## The Two Modes

### Typing Mode
The user signs ASL finger spelling. Each recognized handshape appends a character to a floating text buffer that appears above the current cursor position. When the buffered word or phrase looks correct, a confirm gesture injects it into the active application as simulated keypresses. A delete gesture removes the last character from the buffer. The user never touches the keyboard to type.

### Command Mode
The user performs a gesture bound to a system shortcut — opening an application, running a shell command, triggering a keyboard shortcut, navigating to a URL, or opening a file. Commands fire after a confidence hold window to prevent accidental triggers. The user switches into this mode explicitly and the visual state of the floating buffer reflects which mode is active.

### Mode Switching
A dedicated two-hand attention gesture (both open palms facing camera) toggles between Typing Mode and Command Mode. The floating buffer changes appearance to signal the current mode clearly — the user always knows which mode is active.

---

## Open Source Foundations

### Hand Landmark Extraction — MediaPipe Hands
Google's MediaPipe Hands is the landmark extraction engine. It runs a two-stage pipeline: palm detection on the full frame followed by precise 21-keypoint 3D landmark extraction on the cropped hand region. It tracks between frames so it does not re-detect from scratch every frame, keeping CPU usage low. It is open source under the Apache 2.0 license and installable via pip with no additional dependencies.

All gesture and finger spelling recognition in this project operates on the 63-dimensional normalized landmark vectors that MediaPipe produces, never on raw pixels. This makes classifiers small, fast to train, and robust across different skin tones and backgrounds.

### ASL Finger Spelling Dataset — ASL Alphabet (Kaggle)
The baseline training data for the finger spelling classifier is the ASL Alphabet dataset, publicly available on Kaggle. It contains 87,000 labeled images across 29 classes (26 letters plus space, delete, and nothing). This is the most widely validated public dataset for ASL finger spelling recognition and has been used in numerous published benchmarks.

The agent must not train directly on raw images. Instead, a preprocessing script must run MediaPipe Hands over every image in the dataset, extract the 21 landmark coordinates, and save the resulting landmark vectors as the actual training data. This approach makes the classifier invariant to image background, lighting, and skin tone — it sees only hand geometry, not pixels.

### Static Gesture Classifier — scikit-learn MLPClassifier
A Multi-Layer Perceptron trained on the preprocessed ASL landmark vectors handles finger spelling recognition and static command gestures. scikit-learn's MLPClassifier is the right tool — it is lightweight, fast to train on small datasets, serializes cleanly to disk with pickle, and retrains in seconds when the user adds a new custom gesture. No GPU required.

Target accuracy on the ASL Alphabet dataset after landmark preprocessing is 97% or above per character. Below 95% the error rate compounds across words and the tool becomes frustrating to use.

### Dynamic Gesture Classifier — PyTorch LSTM
Motion-based gestures — the attention/mode-switch gesture, the delete gesture, the confirm gesture, and any user-defined motion shortcuts — require recognizing patterns across time, not single frames. A small LSTM network in PyTorch operates on a rolling buffer of 30 frames of landmark sequences to classify these. PyTorch is used here rather than scikit-learn because sequence modeling requires a recurrent architecture. The LSTM model is kept small enough to run on CPU without meaningful latency.

### Floating Text Buffer — PyQt6 Frameless Overlay
The floating buffer window is a frameless, always-on-top, transparent-background Qt window rendered with PyQt6. It sits above all other windows near the current cursor position and displays the current buffer contents. It has no title bar, no border, and cannot receive focus — it is a passive display element only. It changes color or border style to indicate Typing Mode versus Command Mode.

### Settings Interface — FastAPI + Plain HTML/CSS
A local web server built with FastAPI serves the settings interface at localhost on a fixed port. The frontend is plain HTML and CSS with no JavaScript framework. The interface handles: viewing and editing gesture-to-command bindings, triggering gesture recording sessions, selecting applications or commands to bind, and adjusting system settings. FastAPI is chosen over Flask for its async support and automatic validation — the same server handles real-time communication with the frontend during gesture recording sessions via Server-Sent Events.

### Command Execution — subprocess, pyautogui, keyboard, xdg-open
Four dispatch mechanisms cover all shortcut types. Shell commands use subprocess. Keyboard shortcut simulation uses pyautogui and the keyboard library, which sit on top of the uinput kernel module already present on Linux systems. Application launching and file/URL opening uses xdg-open, the standard Linux utility for opening resources with their default handler. All four are available without root privileges in a normal Linux desktop session.

### Text Injection — pyautogui typewrite
When the user confirms a buffered word or phrase, it is injected into the active application using pyautogui's typewrite function, which simulates individual keystrokes. This works across all applications without requiring accessibility APIs or application-specific integrations.

### Audio Feedback — pyttsx3
All system state communication that does not fit the visual buffer happens through offline text-to-speech via pyttsx3. No network connection required. Key audio events: mode switch confirmation, gesture recording prompts, error states, and first-run setup guidance.

### System Tray — pystray
A pystray tray icon provides passive status visibility with color states mirroring the current mode and confidence state. Right-click menu provides quick access to open settings, toggle the system, and quit.

### Config Persistence — JSON
All gesture bindings, model weights, and user settings persist to disk. Gesture-to-command bindings live in a human-readable JSON config file. MLP weights serialize to pickle. LSTM weights serialize via PyTorch's native save format. The config file hot-reloads when changed on disk using the watchdog library.

---

## Project Structure

```
signtype/
├── main.py                        # Entry point, initializes all threads and services
├── config.json                    # Gesture-to-command bindings and settings
├── requirements.txt
├── README.md
│
├── core/
│   ├── camera.py                  # Webcam capture loop, frame queue
│   ├── landmark_extractor.py      # MediaPipe Hands wrapper, outputs normalized vectors
│   ├── fingerspell_classifier.py  # MLP classifier for ASL letters
│   ├── gesture_classifier.py      # MLP classifier for static command gestures
│   ├── dynamic_classifier.py      # LSTM classifier for motion gestures
│   ├── command_dispatcher.py      # Reads config, executes commands by type
│   ├── text_injector.py           # Manages buffer, confirms and injects text
│   └── state_machine.py           # Manages IDLE / TYPING / COMMAND / RECORDING states
│
├── feedback/
│   ├── buffer_overlay.py          # PyQt6 frameless floating buffer window
│   ├── audio.py                   # pyttsx3 TTS wrapper
│   └── tray.py                    # pystray icon and menu
│
├── training/
│   ├── preprocess_dataset.py      # Runs MediaPipe over ASL Alphabet images, saves landmarks
│   ├── train_fingerspell.py       # Trains MLP on preprocessed ASL landmark data
│   ├── train_dynamic.py           # Trains LSTM on motion gesture sequences
│   ├── recorder.py                # Captures live landmark samples for custom gestures
│   └── trainer.py                 # Retrains MLP/LSTM incrementally after custom recording
│
├── settings/
│   ├── server.py                  # FastAPI server, serves UI and handles recording API
│   ├── static/
│   │   ├── index.html             # Settings interface
│   │   └── style.css
│   └── gesture_store.py           # CRUD for gesture library on disk
│
└── data/
    ├── asl_alphabet_raw/          # Downloaded ASL Alphabet dataset (gitignored)
    ├── landmarks/                 # Preprocessed landmark vectors per gesture class
    ├── custom_gestures/           # User-recorded gesture samples
    ├── model_fingerspell.pkl      # Trained MLP for finger spelling
    ├── model_static.pkl           # Trained MLP for command gestures
    └── model_dynamic.pt           # Trained LSTM for motion gestures
```

---

## State Machine

**IDLE** — Camera runs at low sample rate. No classification. Entered on startup and after a configurable inactivity timeout. Audio announces startup.

**TYPING** — Full landmark extraction every frame. Finger spelling MLP classifies each frame. Characters accumulate in the floating buffer. Delete gesture removes last character. Confirm gesture injects buffer into active application and clears buffer. Motion classifier watches for mode-switch gesture.

**COMMAND** — Full landmark extraction every frame. Static and dynamic command gesture classifiers run. Confidence gating requires 85% confidence held for 500ms before dispatch. 1.5 second cooldown after each command fires. Motion classifier watches for mode-switch gesture.

**RECORDING** — Entered from settings interface or reserved gesture. Audio guides user through performing the target gesture. Landmark samples captured for approximately 5 seconds at full frame rate. On completion, classifier retrains synchronously. Audio confirms completion. Returns to previous mode.

---

## Confidence and Accuracy Rules

These are non-negotiable constraints the agent must enforce throughout:

- Finger spelling MLP must achieve 97% or above per-character accuracy on the preprocessed ASL Alphabet validation split before the model is considered ready for use
- Command gesture confidence threshold is 0.85, hold duration 500ms, cooldown 1.5 seconds
- Finger spelling uses a different confirmation model — a character is appended to the buffer only after it is consistently classified for 300ms, preventing spurious characters during hand transitions between letters
- No gesture fires in IDLE state under any circumstance

---

## Settings Interface Specification

The FastAPI server starts automatically with the main application and listens on localhost port 7842. The settings page is opened from the system tray menu.

The interface must provide these sections:

**Gesture Bindings** — A table of all current custom command gestures showing gesture name, bound command, and command type. Each row has edit and delete actions. An "Add New" button starts the binding creation flow.

**Add Binding Flow** — A form where the user enters a name for the gesture, selects command type (shell / hotkey / launch app / open URL / open file), fills in the command payload using the appropriate input for that type (text field for shell and hotkey, file picker for app and file, URL field for URL), and clicks "Record Gesture" to trigger a recording session. The page shows a live status update via Server-Sent Events while recording proceeds — the camera is capturing, then processing, then done. On completion the binding appears in the table immediately.

**Settings** — Sliders and inputs for confidence threshold, hold duration, cooldown duration, camera index selection, inactivity timeout, and TTS toggle.

**Model Info** — Displays current model accuracy stats, number of custom gestures trained, and a button to retrain from scratch if needed.

---

## Config File Format

```json
{
  "custom_gestures": {
    "gesture_name": {
      "type": "shell | hotkey | launch | url | file",
      "payload": "...",
      "label": "Human readable name"
    }
  },
  "settings": {
    "confidence_threshold": 0.85,
    "hold_duration_ms": 500,
    "cooldown_ms": 1500,
    "fingerspell_hold_ms": 300,
    "camera_index": 0,
    "inactivity_timeout_seconds": 60,
    "tts_enabled": true,
    "settings_port": 7842
  }
}
```

---

## Built-in Gestures

These are trained from the ASL Alphabet dataset and the motion gesture recorder, not user-defined. They are reserved and cannot be overwritten by custom bindings.

**From ASL Alphabet dataset (finger spelling):**
All 26 ASL letter handshapes plus space.

**Motion gestures (LSTM, recorded during first-run setup):**
- Both open palms facing camera — mode switch (TYPING ↔ COMMAND)
- Single downward swipe with index finger — delete last character (TYPING mode)
- Single closed fist hold — confirm and inject buffer (TYPING mode)
- Double open palm — enter IDLE

---

## Threading Model

Seven threads communicate via thread-safe queues. The agent must keep all blocking operations off the main thread.

- **Camera thread** — continuously reads webcam frames, puts onto frame queue
- **Inference thread** — consumes frames, runs landmark extraction and appropriate classifier for current mode, puts gesture events onto event queue
- **State machine thread** — consumes gesture events, manages mode state, routes to appropriate handler
- **Dispatch thread** — handles command confidence gating, cooldown, fires commands
- **Overlay thread** — runs the PyQt6 event loop for the floating buffer window
- **Feedback thread** — handles TTS and tray icon updates asynchronously
- **Settings server thread** — runs FastAPI server, communicates with main app via internal queue for recording triggers and completion events

---

## First Run Experience

On first launch with no trained models present, the system detects this and runs a setup sequence before entering normal operation.

Step 1 — Audio announces first run setup and instructs the user to download the ASL Alphabet dataset from Kaggle and place it in `data/asl_alphabet_raw/`. A URL is printed to the terminal. The system waits.

Step 2 — Once the dataset is detected, preprocessing runs automatically. Audio announces progress. This produces the landmark training data in `data/landmarks/`.

Step 3 — The finger spelling MLP trains on the preprocessed data. Audio announces when complete and reports accuracy. If accuracy is below 97% the agent must log a warning and suggest checking the dataset integrity.

Step 4 — Audio instructs the user to perform each built-in motion gesture (mode switch, delete, confirm, idle) in sequence for recording. The LSTM trains on these samples.

Step 5 — Audio announces the system is ready and explains the two modes briefly. The settings interface URL is announced. Normal operation begins in IDLE mode.

---

## What the Agent Should Build In Order

1. Project structure and requirements.txt with all pinned dependencies
2. Camera capture pipeline — confirm a frame can be read from the webcam
3. MediaPipe Hands integration — confirm landmark extraction produces valid 21-keypoint output
4. Dataset preprocessing script — run MediaPipe over ASL Alphabet images, save landmark vectors
5. Finger spelling MLP — train, validate, confirm 97% accuracy target, serialize to disk
6. State machine skeleton — all four states wired up, transitions working, no classifiers yet
7. Floating buffer overlay — PyQt6 frameless window, mode display, character append and delete
8. Typing Mode — connect inference to buffer, implement fingerspell confidence gating
9. Audio feedback and tray icon
10. Static command gesture MLP — trained on user-recorded samples via recorder module
11. Command Mode — connect static classifier to dispatch, implement confidence gating and cooldown
12. FastAPI settings server — serve UI, implement recording API with SSE status updates
13. LSTM dynamic gesture classifier — mode switch, delete, confirm, custom motion gestures
14. First run setup wizard
15. Config hot-reload, persistence, and model retraining flow

Work incrementally. Each step must be runnable and produce verifiable output before moving to the next. Do not proceed to step N+1 if step N is not confirmed working.

---

## Platform and Licensing Notes

Primary target is Linux (Ubuntu 22.04 and Arch Linux). All dependencies must be installable via pip without root. No cloud APIs. No telemetry. All processing is local and offline.

All chosen open source dependencies are permissively licensed (Apache 2.0, MIT, BSD). The project itself should be released under MIT license. The ASL Alphabet dataset on Kaggle is available under a royalty-free license for non-commercial use — document this clearly in the README and provide instructions for users to download it themselves rather than bundling it.

The architecture must be documented well enough that a contributor can retrain the finger spelling model for BSL, ArSL, or any other sign language by replacing the dataset preprocessing step and retraining, with no changes to the rest of the codebase.
