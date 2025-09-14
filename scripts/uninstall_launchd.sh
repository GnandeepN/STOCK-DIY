#!/usr/bin/env bash
set -euo pipefail

LABEL="com.ai.tradingbot.daily"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

if [[ -f "$PLIST" ]]; then
  launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "[uninstall_launchd] Removed $PLIST"
else
  echo "[uninstall_launchd] No plist at $PLIST"
fi

