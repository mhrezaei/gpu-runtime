#!/usr/bin/env bash
# Runs INSIDE the runtime image, on a CPU, with no GPU and no weights: selfcheck, then every workflow validated on a
# CPU ComfyUI (build_local + validate_local), then the dry run. Called by scripts/gpu-image.sh in a throwaway container
# (build_local writes zero-byte weight placeholders, so never run it in the container you intend to commit).
set -euo pipefail
BIN=/opt/oneart/bin
export OA_HOME="$(mktemp -d)" OA_MODELS="$(mktemp -d)" COMFY_URL=http://127.0.0.1:8190
COMFY_DIR="${COMFY_DIR:-/opt/oneart/ComfyUI}"
export COMFY_DIR

python3 "$BIN/oa.py" --selfcheck --no-gpu

(cd "$COMFY_DIR" && nohup python main.py --cpu --listen 127.0.0.1 --port 8190 --disable-auto-launch >"$OA_HOME/comfy-cpu.log" 2>&1 &)
for _ in $(seq 1 90); do
  curl -fsS "$COMFY_URL/system_stats" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "$COMFY_URL/system_stats" >/dev/null || { tail -n 40 "$OA_HOME/comfy-cpu.log"; echo "CPU ComfyUI did not start"; exit 1; }

mkdir -p "$(dirname "$COMFY_DIR")/api"
python3 "$BIN/build_local.py" "$COMFY_DIR"
python3 -I "$BIN/validate_local.py" "$COMFY_DIR" "$(dirname "$COMFY_DIR")"/api/built_*.json
python3 "$BIN/smoke_plan.py"
python3 "$BIN/dry_run.py"
echo "smoke: ok"
