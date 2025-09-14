#!/usr/bin/env bash
set -euo pipefail

# Installs a launchd job to run the pipeline daily on macOS.
# Defaults: 18:30 local time, Monday–Friday.
# Usage:
#   bash scripts/install_launchd.sh            # install with defaults
#   bash scripts/install_launchd.sh --hour 19 --minute 0           # 19:00
#   bash scripts/install_launchd.sh --interval 7200                # every 2 hours

LABEL="com.ai.tradingbot.daily"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

# Resolve repo directory and script path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT="$REPO_DIR/scripts/run_daily.sh"
PY="$REPO_DIR/venv/bin/python"
LOG_DIR="$REPO_DIR/logs"

mkdir -p "$LOG_DIR"

HOUR=18
MINUTE=30
INTERVAL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hour) HOUR="$2"; shift 2;;
    --minute) MINUTE="$2"; shift 2;;
    --interval) INTERVAL="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# Ensure the runner is executable
chmod +x "$RUN_SCRIPT" || true

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key><string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
      <string>${PY}</string>
      <string>run_all.py</string>
      <string>--parallel</string>
    </array>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/run_all.launchd.out.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/run_all.launchd.err.log</string>
    <key>WorkingDirectory</key>
    <string>${REPO_DIR}</string>
    <key>RunAtLoad</key><true/>
PLIST

if [[ -n "$INTERVAL" ]]; then
  cat >> "$PLIST" <<PLIST
    <key>StartInterval</key><integer>${INTERVAL}</integer>
PLIST
else
  cat >> "$PLIST" <<PLIST
    <key>StartCalendarInterval</key>
    <array>
      <dict><key>Hour</key><integer>${HOUR}</integer><key>Minute</key><integer>${MINUTE}</integer><key>Weekday</key><integer>1</integer></dict>
      <dict><key>Hour</key><integer>${HOUR}</integer><key>Minute</key><integer>${MINUTE}</integer><key>Weekday</key><integer>2</integer></dict>
      <dict><key>Hour</key><integer>${HOUR}</integer><key>Minute</key><integer>${MINUTE}</integer><key>Weekday</key><integer>3</integer></dict>
      <dict><key>Hour</key><integer>${HOUR}</integer><key>Minute</key><integer>${MINUTE}</integer><key>Weekday</key><integer>4</integer></dict>
      <dict><key>Hour</key><integer>${HOUR}</integer><key>Minute</key><integer>${MINUTE}</integer><key>Weekday</key><integer>5</integer></dict>
    </array>
PLIST
fi

cat >> "$PLIST" <<PLIST
  </dict>
</plist>
PLIST

echo "[install_launchd] Wrote: $PLIST"

# Reload the job
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "[install_launchd] Loaded job: ${LABEL}"

# Kick off a run now (optional)
launchctl start "$LABEL" || true
echo "[install_launchd] Started job once for testing. Check logs in $LOG_DIR"
