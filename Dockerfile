# Qwen3.8-27B Uncensored (NVFP4): coding and agent LLM with tool calling
# vLLM on RunPod Serverless, with an OpenAI-compatible API
# All model weights baked into the image.

ARG VLLM_VERSION=0.31.0

# Stage 1: download the weights, check them and split them into four files.
# Only those files reach the final image.
FROM vllm/vllm-openai:v${VLLM_VERSION} AS weights

# pekkAi's NVFP4 build, pinned to a commit. src/sha256sums.txt fails the build
# if any downloaded file differs from that commit.
ARG MODEL_REPO=pekkAi/Qwen3.8-27B-Uncensored-NVFP4-RTX5090
ARG MODEL_REVISION=75082cbabd4af2f823a47f8bce6f3022af28ae1f

# The checkpoint stores its 17.9 GB of weights in a 10.0 GB and a 7.9 GB file.
# reshard.py rewrites them as four files of about 4.5 GB, copying every tensor
# unchanged, because each file becomes one image layer and GHCR rejects layers
# over 10 GB and uploads that take over 10 minutes.
COPY src/download.sh src/sha256sums.txt src/reshard.py /tmp/
RUN bash /tmp/download.sh /tmp/model \
    && cd /tmp/model \
    && sha256sum -c /tmp/sha256sums.txt \
    && python3 /tmp/reshard.py /tmp/model /weights 4 \
    && mv /tmp/model/* /weights/

# Stage 2: the worker.
FROM vllm/vllm-openai:v${VLLM_VERSION}
ARG VLLM_VERSION

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_INPUT=1
ENV PIP_NO_CACHE_DIR=1

# Add the RunPod SDK. vLLM, PyTorch and aiohttp come from the base image. Fail
# the build if installing the SDK replaced the base image's vLLM.
RUN python3 -m ensurepip --upgrade 2>/dev/null || true \
    && python3 -m pip install runpod==1.12.0 \
    && test "$(python3 -c 'import vllm; print(vllm.__version__)')" = "${VLLM_VERSION}"

ENV MODEL_PATH=/models/qwen3.8-27b-uncensored-nvfp4

# One weight file per layer, each about 4.5 GB.
COPY --from=weights /weights/model-00001-of-00004.safetensors ${MODEL_PATH}/
COPY --from=weights /weights/model-00002-of-00004.safetensors ${MODEL_PATH}/
COPY --from=weights /weights/model-00003-of-00004.safetensors ${MODEL_PATH}/
COPY --from=weights /weights/model-00004-of-00004.safetensors ${MODEL_PATH}/
COPY --from=weights /weights/LICENSE /weights/*.json /weights/*.jinja /weights/*.txt ${MODEL_PATH}/

# The weights are in the image, so nothing reaches the Hugging Face Hub at run time.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /

# Add the RunPod handler and startup script.
COPY src/start.sh handler.py test_input.json ./
RUN chmod +x /start.sh

# The base image's entrypoint is `vllm serve`; handler.py starts vLLM itself.
ENTRYPOINT ["/start.sh"]
CMD []
