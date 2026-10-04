#!/usr/bin/env bash
# Run a script inside this project's Isaac Sim container, under this
# account's own home directory -- never another user's install.
#
# Why Docker, not a native/pip install: this machine runs Ubuntu 20.04,
# but current Isaac Sim (pip wheel or standalone) requires Ubuntu 22.04+
# (glibc >= 2.34). Docker sidesteps the host glibc entirely. Image
# version 4.5.0 (not a newer one like 6.1.0) is pinned deliberately:
# newer Isaac Sim's RTX renderer requires NVIDIA driver >= 550.90.07, and
# this machine's driver is 535.230.02 -- 6.1.0 starts but fails to create
# a renderer context; 4.5.0 works cleanly on this driver.
#
# Usage: sim/docker_run.sh <script.py under sim/> [args...]
# Example: sim/docker_run.sh franka_paint_sim.py --headless
#
# Env vars:
#   ROBOT4ART_GPU   which GPU index to use (default: 1 -- check `nvidia-smi`
#                   first and pick an idle one; GPU 0 is often busy with
#                   other users' jobs on this shared workstation).

set -euo pipefail

IMAGE="nvcr.io/nvidia/isaac-sim:4.5.0"
GPU="${ROBOT4ART_GPU:-1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${HOME}/docker/isaac-sim"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <script.py under sim/> [args...]" >&2
  echo "Example: $0 franka_paint_sim.py --headless" >&2
  exit 1
fi

mkdir -p "$CACHE_ROOT"/cache/{kit,ov,pip,glcache,computecache} "$CACHE_ROOT"/{logs,data,documents}

SCRIPT="$1"
shift

docker_args=(
  docker run --rm
  --gpus "device=${GPU}"
  --entrypoint /isaac-sim/python.sh
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y
  -v "${CACHE_ROOT}/cache/kit:/isaac-sim/kit/cache:rw"
  -v "${CACHE_ROOT}/cache/ov:/root/.cache/ov:rw"
  -v "${CACHE_ROOT}/cache/pip:/root/.cache/pip:rw"
  -v "${CACHE_ROOT}/cache/glcache:/root/.cache/nvidia/GLCache:rw"
  -v "${CACHE_ROOT}/cache/computecache:/root/.nv/ComputeCache:rw"
  -v "${CACHE_ROOT}/logs:/root/.nvidia-omniverse/logs:rw"
  -v "${CACHE_ROOT}/data:/root/.local/share/ov/data:rw"
  -v "${CACHE_ROOT}/documents:/root/Documents:rw"
  -v "${REPO_ROOT}/sim:/workspace/sim:rw"
  "${IMAGE}"
  "/workspace/sim/${SCRIPT}"
  "$@"
)

# `sg docker -c '<single command string>'` is needed because this
# account's docker-group membership isn't active in every shell session
# on this machine; build the string safely with printf %q per argument.
cmd_str=""
for a in "${docker_args[@]}"; do
  cmd_str+=" $(printf '%q' "$a")"
done

exec sg docker -c "$cmd_str"
