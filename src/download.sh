#!/usr/bin/env bash
# Downloads $MODEL_REPO at $MODEL_REVISION into the folder given as the only
# argument, leaving out the model card.
set -euo pipefail

# The repo is on Hugging Face's Xet storage. HF_XET_HIGH_PERFORMANCE lets the
# download use every core and all the bandwidth, since RunPod stops
# `docker build` after 30 minutes. HF_HOME is a temporary folder, so no
# Hugging Face cache is left behind.
HF_HOME=/tmp/hf-home HF_XET_HIGH_PERFORMANCE=1 \
    python3 - "$MODEL_REPO" "$MODEL_REVISION" "$1" <<'EOF'
import sys
from huggingface_hub import snapshot_download

snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], ignore_patterns=["README.md", ".gitattributes"])
EOF

rm -rf /tmp/hf-home "$1/.cache"
