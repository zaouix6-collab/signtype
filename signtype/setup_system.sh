#!/bin/bash
# SignType — System dependency installer for Arch Linux
# Run this script once: bash setup_system.sh

set -e

echo "=== SignType System Setup ==="
echo ""

# 1. Install Python 3.12 from AUR
echo "[1/5] Installing Python 3.12 from AUR..."
yay -S --needed python312

# 2. Install system packages
echo "[2/5] Installing system packages..."
sudo pacman -S --needed wtype ydotool gtk4-layer-shell python-gobject gtk4 gobject-introspection

# 3. Enable ydotool daemon
echo "[3/5] Enabling ydotool daemon..."
sudo systemctl enable --now ydotool

# 4. Add user to input group for ydotool non-root access
echo "[4/5] Setting up uinput permissions..."
sudo usermod -aG input $USER

# 5. Create udev rule for uinput
echo "[5/5] Creating udev rule for uinput..."
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/80-uinput.rules
sudo udevadm control --reload-rules
sudo udevadm trigger

echo ""
echo "=== System setup complete ==="
echo "IMPORTANT: Log out and back in for the 'input' group change to take effect."
echo ""
echo "Next steps:"
echo "  cd /home/zaouixm56/Desktop/ig/signtype"
echo "  python3.12 -m venv .venv --system-site-packages"
echo "  source .venv/bin/activate"
echo "  pip install -r requirements.txt"
