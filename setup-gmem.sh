#!/bin/bash
# setup-gmem.sh — Install gmem as a global command on Linux/macOS
# Run this once after setting up the venv

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
GMEM_SCRIPT="$SCRIPT_DIR/gmem.py"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found. Run these first:"
    echo "   python3 -m venv .venv"
    echo "   source .venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Create gmem wrapper in /usr/local/bin (or ~/.local/bin)
TARGET="/usr/local/bin/gmem"
if [ ! -w "$(dirname "$TARGET")" ]; then
    TARGET="$HOME/.local/bin/gmem"
    mkdir -p "$HOME/.local/bin"
fi

cat > "$TARGET" << EOF
#!/bin/bash
exec "$VENV_PYTHON" "$GMEM_SCRIPT" "\$@"
EOF

chmod +x "$TARGET"

echo "✅ gmem command installed at $TARGET"
echo "   Use: gmem --help"
echo ""
echo "   If not on PATH, add: export PATH=\"$(dirname "$TARGET"):\$PATH\""
