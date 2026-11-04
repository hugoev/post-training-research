# Source before training or vLLM runs: `source scripts/env.sh`.
# Applies workarounds only when the proper toolchain is missing (e.g. WSL without
# build-essential or the CUDA toolkit). With gcc and nvcc installed, this is a no-op.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Long runs are logged to files; without this, metric prints lag far behind the progress bars.
export PYTHONUNBUFFERED=1

if ! command -v cc >/dev/null 2>&1 && [ -z "${CC:-}" ]; then
    # Triton JIT-compiles a small C shim; Zig's bundled clang works without root.
    if [ ! -x "$repo_root/.tools/cc" ]; then
        uv pip install -q ziglang
        mkdir -p "$repo_root/.tools"
        printf '#!/bin/sh\nexec "%s" -m ziglang cc "$@"\n' "$repo_root/.venv/bin/python" \
            > "$repo_root/.tools/cc"
        chmod +x "$repo_root/.tools/cc"
    fi
    export CC="$repo_root/.tools/cc"
fi

if [ -d /usr/lib/wsl/lib ]; then
    export TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib
fi

if ! command -v nvcc >/dev/null 2>&1 && [ ! -d /usr/local/cuda ]; then
    # FlashInfer's sampler is JIT-compiled with nvcc; fall back to vLLM's PyTorch sampler.
    export VLLM_USE_FLASHINFER_SAMPLER=0
fi
