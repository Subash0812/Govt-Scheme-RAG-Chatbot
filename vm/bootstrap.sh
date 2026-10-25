#!/usr/bin/env bash
# Bootstrap the SchemeSathi development VM (Ubuntu 24.04, no display).
#
# Run as the normal sudo-capable user, from anywhere:
#     bash ~/bootstrap.sh
#
# Idempotent: safe to rerun. It does NOT need the Gemini key; copy .env in
# separately (scp) before running stage 3 or anything that calls the API.

set -euo pipefail

REPO_URL="https://github.com/Subash0812/Govt-Scheme-RAG-Chatbot.git"
PROJECT_DIR="${PROJECT_DIR:-$HOME/schemebot}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() { printf '\n=== %s ===\n' "$*"; }

log "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
  git tmux curl ca-certificates \
  python3 python3-venv python3-dev \
  build-essential libgomp1 rsync

log "Python version"
"$PYTHON_BIN" --version

log "Clone (shallow: the full history carries ~400 MB of an old .venv)"
if [ ! -d "$PROJECT_DIR/.git" ]; then
  git clone --depth 1 "$REPO_URL" "$PROJECT_DIR"
else
  echo "$PROJECT_DIR already exists, skipping clone"
fi

cd "$PROJECT_DIR"

log "Virtualenv + dependencies"
if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip wheel
pip install --quiet -r requirements.txt

log "Unit tests (no model download; uses a fake encoder)"
python -m unittest discover -s tests -v

log "Embedding model"
# Stage 5 is the ONLY module that downloads. Everything else loads with
# local_files_only=True from data/models, so this must run before evals.
if [ -f data/vector_store/bge_base_en_v1_5/index.faiss ]; then
  echo "chunk index present, skipping rebuild"
else
  python -m src.embeddings.build_faiss_index
  python -m src.embeddings.build_scheme_card_index
fi

log "Verify index fingerprint"
python - <<'PY'
import json, pathlib
p = pathlib.Path("data/vector_store/bge_base_en_v1_5/scheme_manifest.json")
if p.exists():
    m = json.loads(p.read_text())
    print("scheme cards sha256:", m.get("cards_sha256"))
    print("expected          : e2785e6235d82343ba6398ae3f45c630301e67fce4ccd9d1ac7282d7d7549c74")
else:
    print("no scheme_manifest.json yet")
PY

log "Qoder CLI"
if command -v qoder >/dev/null 2>&1; then
  qoder --version || true
else
  curl -fsSL https://qoder.com/install | bash
  echo "Installed. Run 'qoder' once to authenticate, or export QODER_PERSONAL_ACCESS_TOKEN."
fi

cat <<'EOF'

------------------------------------------------------------------------
Bootstrap finished. Remaining manual steps:

1. Copy secrets (from the old machine, never through git):
     scp .env <vm>:~/schemebot/.env

2. Authenticate Qoder CLI, then start the phone-reachable daemon:
     cd ~/schemebot
     tmux new -s qoder
     qoder remote-control

3. Optional: install the daemon as a service so it survives reboots:
     sudo bash vm/install-remote-control-service.sh

4. Stop the VM when you are not working; see PROJECT_HANDOFF.txt section 11.
------------------------------------------------------------------------
EOF
