# Qwen3.8-27B Uncensored (NVFP4) on RunPod Serverless

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with its refusals removed ("abliterated") by OrcaRouter, quantized to NVFP4 (4-bit) by pekkAi, running in vLLM as a RunPod serverless worker. It is a coding and agent model: it calls tools, thinks before it answers, takes up to 262,144 tokens of context and can read images. The endpoint speaks the OpenAI API, so coding agents and OpenAI client libraries can use it directly.

This is the same worker as [runpod_Qwen3.8_27B](https://github.com/Akshay7623/runpod_Qwen3.8_27B), the FP8 build, with different weights. The handler, the requests and the model name are the same, so a client of one endpoint works with the other unchanged. See [Differences from the FP8 worker](#differences-from-the-fp8-worker).

All model files are baked into the Docker image, so the endpoint needs no network volume and no Hugging Face token.

> **This model has no safety refusals.** It answers requests the original Qwen3.8-27B refuses, including harmful and illegal ones. Anyone with the endpoint's API key gets that, so keep the key private. See [License](#license).

## What is in the image

| Part | What | Size |
| --- | --- | --- |
| Model weights | Qwen3.8-27B Uncensored: 27B dense, hybrid attention, NVFP4 weights and activations (group size 16), NVFP4 `lm_head`, BF16 vision encoder, no MTP head | 17.9 GB |
| Inference server | vLLM 0.31.0, CUDA 13.0 build (`vllm/vllm-openai:v0.31.0`) | 9.0 GB download |
| Worker | RunPod SDK 1.12.0 and [handler.py](handler.py) | |

[handler.py](handler.py) starts `vllm serve` inside the container, waits until the model is loaded, then forwards each job to vLLM's OpenAI API. If vLLM crashes later, the worker stops so RunPod replaces it.

### Differences from the FP8 worker

| | FP8 worker | This worker |
| --- | --- | --- |
| Weights | OrcaRouter's block FP8, 30.9 GB | pekkAi's NVFP4, 17.9 GB |
| GPU | 48 GB (L40S, RTX 6000 Ada) or H100 / H200 | 32 GB Blackwell (RTX 5090) |
| MTP speculative decoding | On (`MTP_TOKENS`, default 3) | Not available: pekkAi removed the MTP head. `MTP_TOKENS` is gone. |
| `MAX_CONCURRENCY` | Jobs per worker | Jobs per worker, and also vLLM's `--max-num-seqs` |

NVFP4 stores weights in 4 bits instead of 8, so it can lose more accuracy than FP8. pekkAi's model card gives one figure: 92% on 50 MMMU-Pro questions. That is too few questions to compare the two builds, and it was not checked here.

### Where the weights come from

[pekkAi/Qwen3.8-27B-Uncensored-NVFP4-RTX5090](https://huggingface.co/pekkAi/Qwen3.8-27B-Uncensored-NVFP4-RTX5090) is not gated, so the build downloads it without a token. It is pekkAi's own quantization of OrcaRouter's BF16 build, [orcarouter/Qwen3.8-27B-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored), made with NVIDIA Model Optimizer 0.47.0rc0. It is not an OrcaRouter or NVIDIA release.

The Dockerfile pins the repo to commit `75082cbabd4af2f823a47f8bce6f3022af28ae1f`, so its content cannot change. [src/sha256sums.txt](src/sha256sums.txt) holds the SHA-256 of all 15 files the build downloads, and the build fails if any of them does not match. The hashes were checked on 2026-10-17 against Hugging Face's file listing for that commit. `README.md` and `.gitattributes` are not downloaded.

Seven of the files are byte-identical to OrcaRouter's FP8 release, which the FP8 worker uses: `tokenizer.json`, `vocab.json`, `merges.txt`, `chat_template.jinja`, `preprocessor_config.json`, `video_preprocessor_config.json` and `LICENSE`. Prompts are therefore tokenized and formatted exactly as they are there.

How pekkAi's checkpoint differs from a standard Model Optimizer NVFP4 export, from its model card:

- `lm_head` is in NVFP4 rather than BF16 (0.67 GiB instead of 2.37 GiB).
- The MTP head is removed (0.79 GiB), and `config.json` sets `mtp_num_hidden_layers` to 0.
- The vision encoder, the embeddings and the small linear-attention projections stay in BF16.
- Calibration used 512 image-text samples.

If pekkAi deletes the repo, rebuilds fail until `MODEL_REPO` and `MODEL_REVISION` in the Dockerfile point somewhere else. Images that are already built keep working.

### Why the weight files are re-split

pekkAi stores the weights in two files, of 10.0 GB and 7.9 GB. Each file in the image becomes one layer, and GHCR, where the build pushes the image, rejects layers over 10 GB and uploads that take over 10 minutes. The FP8 worker's push took about 15 minutes for layers of 4.4 GB, so a 10 GB layer is likely to fail.

The Dockerfile therefore has two stages. The first downloads the checkpoint, checks it against [src/sha256sums.txt](src/sha256sums.txt), and runs [src/reshard.py](src/reshard.py), which writes the same 2,387 tensors into four files of about 4.5 GB and writes a new `model.safetensors.index.json`. Each tensor's bytes are copied unchanged. The script then reads every new file back with the safetensors library and fails the build if any tensor's dtype, shape or SHA-256 differs from the original. The second stage copies the four files into the image, one layer each.

So the weight files and the index in the image do not have the hashes Hugging Face lists for pekkAi's repo, but every tensor in them does.

## Deploy

GitHub Actions builds the image ([build-image.yml](.github/workflows/build-image.yml)) and pushes it to GHCR as `ghcr.io/akshay7623/runpod_qwen3.8_27b_nvfp4`; nothing is built locally. A build runs on every push to `main` that changes the Dockerfile, `handler.py`, `test_input.json` or `src/`, and can also be started by hand from the repo's **Actions** tab. Each image is tagged `latest` and `sha-` plus the first 7 characters of the commit, and the run's summary shows the tag.

If this repo is public, the package becomes public on the first push, as the FP8 worker's did, so anyone can pull the image. To make it private, open the package on GitHub (**Packages → runpod_qwen3.8_27b_nvfp4 → Package settings → Change visibility**). A private package needs a token for RunPod to pull it:

1. On GitHub, create a classic personal access token with only the `read:packages` scope (**Settings → Developer settings → Personal access tokens → Tokens (classic)**).
2. In the RunPod console, under **Settings → Container Registry Auth**, add a credential with username `Akshay7623` and the token as the password.
3. Go to **Serverless → New Endpoint**, choose a Docker image, enter `ghcr.io/akshay7623/runpod_qwen3.8_27b_nvfp4:sha-<commit>`, and pick that credential.
4. Set the endpoint options below, then deploy.

To update the endpoint, change its image to the new `sha-` tag. Use those tags rather than `latest`, so the endpoint only changes when you change the tag, and workers do not keep running an older `latest` they have cached.

| Setting | Value | Why |
| --- | --- | --- |
| Allowed CUDA Versions | `13.0` | vLLM's default image is built for CUDA 13.0 and needs NVIDIA driver 580 or newer. On an older host the worker exits at startup and says so. |
| GPU | 32 GB PRO (RTX 5090) | The checkpoint is made for it. The weights take 17.9 GB and the rest holds the KV cache. Blackwell cards run NVFP4 in hardware; larger ones (RTX PRO 6000, B200) do too and have not been tried. Older GPUs have no FP4 hardware; vLLM has slower fallback kernels for them, which have not been tried with this checkpoint. |
| FlashBoot | On | Without it, every new worker loads the model from scratch before its first job. |
| Execution timeout | `1800` s to start | The default is 600 s, and a reply with long thinking can run past it. |
| Max workers | 1 while testing | |

A new worker loads the weights onto the GPU, and vLLM compiles the model and records CUDA graphs before the first job runs. This has not been timed on this image. Jobs wait in the queue until the worker is ready.

Environment variables, all optional:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MAX_MODEL_LEN` | `131072` | Longest prompt plus reply, in tokens. The model's native limit is `262144`. See [Context length](#context-length). |
| `MAX_CONCURRENCY` | `8` | Jobs one worker runs at the same time, and vLLM's `--max-num-seqs`. vLLM batches them on the GPU, so they share its speed and its KV cache. |
| `GPU_MEMORY_UTILIZATION` | `0.92` | Share of GPU memory vLLM may use. |
| `KV_CACHE_DTYPE` | `fp8` | KV cache precision. `auto` stores it in BF16, which halves the context that fits. |
| `SERVED_MODEL_NAME` | `qwen3.8-27b` | Model name in responses. Requests can send any `model`; the worker replaces it with this one. |
| `EXTRA_VLLM_ARGS` | | More `vllm serve` flags, with shell-style quoting. For example, `--language-model-only` drops the vision encoder to free memory for KV cache. |
| `VLLM_STARTUP_TIMEOUT` | `1200` | Seconds to wait for vLLM to load before the worker gives up. |
| `REQUEST_TIMEOUT` | `3600` | Seconds a single request to vLLM may take. |

### The GitHub Actions build

The build needs about 65 GB of disk: the 23 GB base image, the weights in the first stage, and the same weights again in the final image. That is more than a standard GitHub runner has free. The workflow first deletes the runner's preinstalled SDKs, then joins the free space on its two disks into one volume for Docker. The pushed image is about 27 GB to download (9 GB base image and 18 GB of weights).

### RunPod's GitHub build

The same Dockerfile also builds with RunPod's own GitHub integration (**Serverless → New Endpoint → Import Git Repository**, branch `main`, Dockerfile path `Dockerfile`). There, every push to `main` starts a build. Those builds are limited to an 80 GB image, 160 minutes in total, and 30 minutes for the `docker build` step.

On 2026-10-16 that route failed three times in a row for the FP8 worker, on RunPod's side, which is why the GitHub Actions build exists. The log stopped at `[internal] booting buildkit` with `OCI runtime create failed ... unsafe procfs detected ... invalid cross-device link`, before RunPod had read the Dockerfile.

## Requests

### OpenAI-compatible URL

Coding agents and OpenAI client libraries need three values:

| | |
| --- | --- |
| Base URL | `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1` |
| API key | Your RunPod API key |
| Model | `qwen3.8-27b` (any name works) |

```bash
curl https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/chat/completions \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Write a binary search in Rust."}], "max_tokens": 8192}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1", api_key="<RUNPOD_API_KEY>")
stream = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "Write a binary search in Rust."}],
    max_tokens=8192,
    stream=True,
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Tools, `tool_choice` and streaming work as in the OpenAI API, and tool calls come back as standard `tool_calls`. RunPod hands these requests to the worker as `{"openai_route": ..., "openai_input": ...}`, the same way it does for its own vLLM worker, and the handler passes them to vLLM unchanged. Only `/v1/` routes are accepted.

### /run and /runsync

`input` is an OpenAI chat completions body, or a text completions body with `prompt` instead of `messages`:

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

The output is a list, because the handler is able to stream. Without `stream`, the list holds one chat completion:

```json
{
  "status": "COMPLETED",
  "output": [
    {
      "choices": [{ "message": { "role": "assistant", "reasoning": "...", "content": "..." }, "finish_reason": "stop" }],
      "usage": { "prompt_tokens": "...", "completion_tokens": "...", "total_tokens": "..." }
    }
  ]
}
```

With `"stream": true`, send the job to `/run` and read `/stream/<job_id>`. Each item is one chunk, with the new text in `choices[0].delta`.

When a request fails, the job is marked FAILED and `error` holds vLLM's message, for example when `max_tokens` is larger than the context allows.

Unlike the ComfyUI workers, this one has no S3 upload and no Lambda callbacks: the reply comes back in the job output.

Three ready-made requests are in the repo. They take the place of the ComfyUI workers' workflow files, because a language model has no graph; `input` is the request body itself.

- [test_input.json](test_input.json): a short coding question with thinking off. Small and quick, for a first test.
- [sample-req-chat.json](sample-req-chat.json): a debugging question with thinking on, using Qwen's sampling settings for thinking mode.
- [sample-req-tools.json](sample-req-tools.json): one step of a coding-agent loop. It defines three tools and contains a failed test run that `run_command` returned. The reply should be the next tool call, most likely `read_file`.

### Thinking

Thinking is on by default. It comes back in the message's `reasoning` field (`delta.reasoning` when streaming), and the answer in `content`.

| To | Add to the request |
| --- | --- |
| Turn thinking off | `"chat_template_kwargs": {"enable_thinking": false}` |
| Think less | `"reasoning_effort": "medium"` or `"low"`. The default is `"xhigh"`; other values are rejected. |
| Keep only the latest turn's thinking | `"chat_template_kwargs": {"preserve_thinking": false}` |

By default, thinking from earlier assistant turns stays in the prompt when the client sends it back in each message's `reasoning` field. Qwen recommends this for agents, since the model does not have to work out the same things again.

When a request leaves out sampling settings, vLLM uses the model's own defaults: `temperature` 1.0, `top_p` 0.95, `top_k` 20. Those are Qwen's settings for thinking mode. With thinking off, Qwen recommends `temperature` 0.7, `top_p` 0.8 and `presence_penalty` 1.5, as in [test_input.json](test_input.json).

Thinking can take thousands of tokens before the answer starts, so a small `max_tokens` can end a reply before it has any `content`.

### Images

The model also reads images sent as `image_url` parts of a message, as base64 `data:` URLs. RunPod limits the size of a request, so scale large images down first.

## Context length

`MAX_MODEL_LEN` sets the context, up to the native 262,144 tokens. Only 16 of the model's 64 layers keep a KV cache; the others use linear attention with a fixed-size state. With the FP8 KV cache that is about 32 KiB per token, so long contexts need much less memory than in most models this size.

How much fits on an RTX 5090 has not been measured with this checkpoint. Two figures from other people, for comparison:

- [gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090), the censored model with the same layout (NVFP4 `lm_head`, no MTP head), reports that vLLM 0.27 on an RTX 5090 holds about 205,000 tokens of KV cache at 0.90 GPU memory utilization, and the full 262,144 at 0.97.
- pekkAi reports that this checkpoint was not reliable at 200,000 tokens of context but ran stably at 160,000, in SGLang.

vLLM prints the real figure at startup (`GPU KV cache size: ... tokens`). If `MAX_MODEL_LEN` does not fit, vLLM stops at startup and the worker log says why. To make room for more context, raise `GPU_MEMORY_UTILIZATION`, or add `--language-model-only` to `EXTRA_VLLM_ARGS` if you do not need images.

## Speculative decoding

The FP8 worker drafts tokens with the model's own MTP head. This checkpoint has none, so this worker generates one token per step. pekkAi recommends a separate drafter model, [gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4), with SGLang. This worker runs vLLM and does not include it.

## License

The weights are under the [Apache License 2.0](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE), from Qwen3.8-27B through OrcaRouter's build and pekkAi's quantization. It allows commercial use and hosting. The license file ships in the image next to the weights.

The license does not restrict what the model is used for, but OrcaRouter's card is direct about what the abliteration does: the model "will comply with harmful, unethical, offensive, or illegal requests that the original Qwen3.8-27B would refuse". OrcaRouter releases it for research such as red-teaming and studying refusals. Laws and RunPod's terms still apply, and whoever runs the endpoint is responsible for what it produces. If other people can reach it, put your own checks in front of it.

OrcaRouter's figures for refusal rates and pekkAi's accuracy figure are their own and were not checked here.

vLLM is under the Apache License 2.0. The worker code in this repo has no license file yet.
