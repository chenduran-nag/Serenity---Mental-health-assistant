#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  export NVIDIA_VISIBLE_DEVICES=all
  export NVIDIA_DRIVER_CAPABILITIES=compute,utility
  DEVICE_MODE="GPU"
else
  export NVIDIA_VISIBLE_DEVICES=none
  export NVIDIA_DRIVER_CAPABILITIES=utility
  DEVICE_MODE="CPU"
fi

cat <<'EOF'
  _____ ______ _____  ______ _   _ _____ _________     __
 / ____|  ____|  __ \|  ____| \ | |_   _|__   __\ \   / /
| (___ | |__  | |__) | |__  |  \| | | |    | |   \ \_/ /
 \___ \|  __| |  _  /|  __| | . ` | | |    | |    \   /
 ____) | |____| | \ \| |____| |\  |_| |_   | |     | |
|_____/|______|_|  \_\______|_| \_|_____|  |_|     |_|
EOF

echo
echo "Device mode: ${DEVICE_MODE}"
echo "Frontend: http://localhost:${SERENITY_FRONTEND_PORT:-5173}"
echo "Backend:  http://localhost:${SERENITY_BACKEND_PORT:-8000}"
echo "Model:    http://localhost:${SERENITY_MODEL_PORT:-8001}"
echo "STT:      http://localhost:${SERENITY_STT_PORT:-8002}"
echo "TTS:      http://localhost:${SERENITY_TTS_PORT:-8003}"
echo "VAD:      ws://localhost:${SERENITY_VAD_PORT:-8004}/vad"
echo

docker compose -f docker-compose.yml up --build
