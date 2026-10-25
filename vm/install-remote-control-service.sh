#!/usr/bin/env bash
# Install the Qoder remote-control daemon as a systemd service so the phone can
# dispatch tasks without an SSH session holding it open.
#
#     sudo bash vm/install-remote-control-service.sh
#
# Re-run after changing the project path. Remove with:
#     sudo systemctl disable --now qoder-remote

set -euo pipefail

RUN_USER="${SUDO_USER:-$USER}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
QODER_BIN="${QODER_BIN:-$(sudo -u "$RUN_USER" bash -lc 'command -v qoder' || true)}"
UNIT=/etc/systemd/system/qoder-remote.service

if [ -z "$QODER_BIN" ]; then
  echo "Could not find the 'qoder' binary on PATH for $RUN_USER." >&2
  echo "Install it first (see vm/bootstrap.sh), or set QODER_BIN=/path/to/qoder." >&2
  exit 1
fi

cat > "$UNIT" <<EOF
[Unit]
Description=Qoder remote-control daemon (SchemeSathi dev box)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$QODER_BIN remote-control
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now qoder-remote
systemctl --no-pager --lines=20 status qoder-remote || true
