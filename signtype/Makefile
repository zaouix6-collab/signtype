.PHONY: setup train run clean

# System deps (Arch Linux — requires sudo)
system-deps:
	yay -S --noconfirm python312
	sudo pacman -S --needed wtype ydotool gtk4-layer-shell python-gobject gtk4 gobject-introspection
	sudo systemctl enable --now ydotool

# Python virtual environment
venv:
	python3.12 -m venv .venv --system-site-packages
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip freeze > requirements.lock
	@echo "✓ Virtual environment ready. Activate with: source .venv/bin/activate"

# Download ASL dataset via Kaggle CLI
dataset:
	@if [ -z "$$KAGGLE_API_TOKEN" ] && [ -f ../.env ]; then \
		export $$(cat ../.env | xargs); \
	fi; \
	mkdir -p data/asl_alphabet_raw; \
	/tmp/kaggle-env/bin/kaggle datasets download grassknoted/asl-alphabet \
		-p data/asl_alphabet_raw --unzip || \
	echo "Set KAGGLE_API_TOKEN or download manually from Kaggle"

# Preprocess + train
train: train-fingerspell

train-fingerspell:
	.venv/bin/python training/preprocess_dataset.py
	.venv/bin/python training/train_fingerspell.py

train-dynamic:
	.venv/bin/python training/train_dynamic.py

# Run SignType
run:
	.venv/bin/python main.py

# Record a custom gesture (usage: make record NAME=wave TYPE=static)
record:
	.venv/bin/python training/recorder.py $(NAME) --type $(TYPE)

# Retrain after recording new gestures
retrain:
	.venv/bin/python training/trainer.py

# Full setup
setup: venv dataset train
	@echo "✓ Setup complete! Run 'make run' to start SignType."

# Clean generated files
clean:
	rm -rf data/landmarks/ data/*.pkl data/*.pt
	rm -rf __pycache__ core/__pycache__ feedback/__pycache__ training/__pycache__ settings/__pycache__

# Push to GitHub
push:
	cd .. && git add -A && git commit -m "Update SignType" && git push origin main
