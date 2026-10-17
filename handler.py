"""RunPod serverless handler for Qwen3.8-27B Uncensored (NVFP4) on vLLM.

At startup the handler launches `vllm serve` on 127.0.0.1, waits until the
model is loaded, and only then starts taking jobs. Each job is forwarded to
vLLM's OpenAI-compatible API; this file never imports vLLM itself.

A job's `input` can take three shapes:

1. What RunPod sends for requests made to the endpoint's /openai/v1 URL:
       {"openai_route": "/v1/chat/completions", "openai_input": {...}}
2. A chat completions request body:
       {"messages": [...], "max_tokens": 2048, "stream": false, ...}
3. A text completions request body:
       {"prompt": "...", "max_tokens": 256, ...}
"""

import atexit
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

import aiohttp
import runpod

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/qwen3.8-27b-uncensored-nvfp4")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen3.8-27b")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "131072")
GPU_MEMORY_UTILIZATION = os.environ.get("GPU_MEMORY_UTILIZATION", "0.92")
KV_CACHE_DTYPE = os.environ.get("KV_CACHE_DTYPE", "fp8")
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "8"))
EXTRA_VLLM_ARGS = os.environ.get("EXTRA_VLLM_ARGS", "")
STARTUP_TIMEOUT_S = int(os.environ.get("VLLM_STARTUP_TIMEOUT", "1200"))
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT", "3600"))

VLLM_PORT = 8000
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}"

shutting_down = threading.Event()


def vllm_command():
    """The `vllm serve` command line, built from the environment."""
    cmd = [
        "vllm", "serve", MODEL_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        "--kv-cache-dtype", KV_CACHE_DTYPE,
        # The worker never sends vLLM more than MAX_CONCURRENCY requests at once.
        # A lower limit also leaves more of a 32 GB card for the KV cache, and
        # vLLM records fewer CUDA graphs at startup.
        "--max-num-seqs", str(MAX_CONCURRENCY),
        # Puts the model's thinking in the message's `reasoning` field instead
        # of mixing it into `content`.
        "--reasoning-parser", "qwen3",
        # Qwen3.8 writes tool calls as <tool_call><function=...> XML; this parser
        # returns them as standard OpenAI `tool_calls`.
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]
    return cmd + shlex.split(EXTRA_VLLM_ARGS)


def start_vllm():
    """Launch vLLM and block until its /health answers 200."""
    cmd = vllm_command()
    print(f"worker-vllm - Starting vLLM: {shlex.join(cmd)}", flush=True)
    # vLLM's output goes straight to the worker log.
    proc = subprocess.Popen(cmd)

    started = time.monotonic()
    while time.monotonic() - started < STARTUP_TIMEOUT_S:
        if proc.poll() is not None:
            sys.exit(f"worker-vllm - vLLM exited during startup with code {proc.returncode}. The reason is in its log above.")
        try:
            with urllib.request.urlopen(f"{VLLM_URL}/health", timeout=5) as resp:
                if resp.status == 200:
                    print(f"worker-vllm - vLLM is ready after {time.monotonic() - started:.0f} s", flush=True)
                    return proc
        except OSError:
            pass
        time.sleep(2)

    proc.kill()
    sys.exit(f"worker-vllm - vLLM was not ready after {STARTUP_TIMEOUT_S} s (VLLM_STARTUP_TIMEOUT).")


def exit_when_vllm_dies(proc):
    """Stop the worker if vLLM crashes, so RunPod replaces it instead of
    sending it jobs it can no longer run."""
    code = proc.wait()
    if not shutting_down.is_set():
        print(f"worker-vllm - vLLM exited with code {code}. Stopping the worker.", flush=True)
        os._exit(1)


def stop_vllm(proc):
    shutting_down.set()
    proc.terminate()


def error_body(message):
    """A failed job. RunPod marks a job FAILED when the handler yields a dict
    with an `error` key, and keeps the error as a string."""
    return {"error": message}


def parse_input(job_input):
    """Return (method, route, body) for a job's input."""
    if not isinstance(job_input, dict):
        raise ValueError("`input` must be a JSON object.")

    if "openai_route" in job_input:
        route = job_input["openai_route"]
        # Only the OpenAI API is forwarded, and only to the local vLLM.
        if not isinstance(route, str) or not route.startswith("/v1/"):
            raise ValueError(f"Unsupported openai_route: {route!r}")
        body = job_input.get("openai_input")
        return ("POST" if body is not None else "GET"), route, body

    if "messages" in job_input:
        return "POST", "/v1/chat/completions", job_input
    if "prompt" in job_input:
        return "POST", "/v1/completions", job_input

    raise ValueError("`input` needs `messages` (chat), `prompt` (text completion) or `openai_route`.")


async def vllm_error(resp):
    """A failed job carrying vLLM's own error message."""
    text = await resp.text()
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        body = None
    if isinstance(body, dict):
        # Recent vLLM nests the message under `error`; older versions do not.
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        text = err.get("message", text)
    return error_body(f"vLLM returned HTTP {resp.status}: {text}")


async def sse_events(resp):
    """Yield each complete server-sent event ("data: ...\\n\\n") from vLLM's stream."""
    buffer = b""
    async for chunk in resp.content.iter_any():
        buffer += chunk
        while b"\n\n" in buffer:
            event, buffer = buffer.split(b"\n\n", 1)
            yield event.decode("utf-8") + "\n\n"
    if buffer.strip():
        yield buffer.decode("utf-8")


async def handler(job):
    job_input = job.get("input")
    try:
        method, route, body = parse_input(job_input)
    except ValueError as e:
        yield error_body(str(e))
        return

    if isinstance(body, dict):
        # The worker serves one model, so any model name the client sends means this one.
        body = {**body, "model": SERVED_MODEL_NAME}
    stream = isinstance(body, dict) and body.get("stream") is True
    # RunPod's /openai/v1 URL passes the stream on to the client as is, so it
    # gets vLLM's raw server-sent events. Other jobs get one JSON object per chunk.
    raw_events = "openai_route" in job_input

    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, VLLM_URL + route, json=body) as resp:
                if resp.status >= 400:
                    yield await vllm_error(resp)
                    return

                if not stream:
                    yield await resp.json(content_type=None)
                    return

                async for event in sse_events(resp):
                    if raw_events:
                        yield event
                        continue
                    for line in event.splitlines():
                        if line.startswith("data:") and line[5:].strip() != "[DONE]":
                            yield json.loads(line[5:])
    except (aiohttp.ClientError, TimeoutError) as e:
        print(f"worker-vllm - Request to vLLM failed: {e!r}", flush=True)
        yield error_body(f"Request to vLLM failed: {e!r}")


def main():
    vllm_process = start_vllm()
    atexit.register(stop_vllm, vllm_process)
    threading.Thread(target=exit_when_vllm_dies, args=(vllm_process,), daemon=True).start()

    print("worker-vllm - Starting RunPod handler", flush=True)
    runpod.serverless.start(
        {
            "handler": handler,
            # vLLM batches requests on the GPU, so one worker takes several jobs at once.
            "concurrency_modifier": lambda current: MAX_CONCURRENCY,
            # With /run and /runsync, the job output is the list of everything yielded.
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    main()
