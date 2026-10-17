#!/usr/bin/env bash

# This image ships vLLM's CUDA 13.0 build, which needs NVIDIA driver >= 580. On
# an older host vLLM would fail deep inside its startup, so check here and fail
# with the real cause.
echo "worker-vllm: Checking GPU availability..."
if ! GPU_CHECK=$(python3 -c "
import torch
try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    _ = (torch.zeros(8, device='cuda') + 1).sum().item()
    torch.cuda.synchronize()
    print(f'OK: {name} (sm_{major}{minor}), torch {torch.__version__}, cuda {torch.version.cuda}')
except Exception as e:
    print(f'FAIL: {e}')
    exit(1)
" 2>&1); then
    echo "worker-vllm: GPU is not usable with this PyTorch build: $GPU_CHECK"
    echo "worker-vllm: Set 'Allowed CUDA Versions' to 13.0 on the endpoint so workers get a host with driver >= 580."
    exit 1
fi
echo "worker-vllm: GPU available - $GPU_CHECK"

# handler.py starts vLLM, waits for the model to load, then takes jobs.
# Serve the RunPod API locally for testing, or join the endpoint's job queue.
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    exec python3 -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    exec python3 -u /handler.py
fi
