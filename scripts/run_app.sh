#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIMARY_STREAMLIT="$ROOT_DIR/.venv/bin/streamlit"
LEGACY_STREAMLIT="$ROOT_DIR/companion-agent/bin/streamlit"

cd "$ROOT_DIR"

if [[ -x "$PRIMARY_STREAMLIT" ]]; then
  exec "$PRIMARY_STREAMLIT" run app.py
fi

if [[ -x "$LEGACY_STREAMLIT" ]]; then
  exec "$LEGACY_STREAMLIT" run app.py
fi

exec streamlit run app.py
