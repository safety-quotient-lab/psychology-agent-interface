#!/usr/bin/env python3
"""
llm-serve.py — Persistent LLM sidecar for pai (Go TUI).

Loads a model once, keeps it in VRAM, and serves inference requests
via newline-delimited JSON on stdin/stdout.

Protocol:
  Startup: writes {"status":"ready","model":"...","vram_mb":N,"load_s":N.N,"use_native":bool}
  Request: {"messages":[...],"max_tokens":N,"temperature":N.N}
  Response: {"reply":"...","tokens":N,"seconds":N.N}
  Error:    {"error":"..."}
"""

import sys, os, time, json, argparse

# Silence transformers/torch noise
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _detect_device():
    """Return 'cuda', 'mps', or 'cpu' depending on available hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE = _detect_device()


def _empty_cache():
    """Release unused memory on the current device."""
    try:
        import torch
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass

# Load HF token from ~/.dev.vars
_devvars = os.path.expanduser("~/.dev.vars")
if os.path.exists(_devvars):
    for _line in open(_devvars):
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.strip().partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

def _load_catalog():
    """Load model catalog from models.json (single source of truth)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    catalog_path = os.path.join(script_dir, "..", "models.json")
    with open(catalog_path) as f:
        entries = json.load(f)
    # Convert to dict keyed by model key: (huggingface_id, dtype, gated)
    return {
        e["key"]: (e["huggingface"], e["dtype"], e["gated"])
        for e in entries
    }, {e["key"]: e for e in entries}

MODELS, MODEL_CATALOG = _load_catalog()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Execute a bash command. Returns combined stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string", "description": "Bash command to run"}},
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Absolute or relative file path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (overwrites if it exists). Use edit_file instead for targeted changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace the first occurrence of old_str with new_str in a file. old_str must match exactly (whitespace matters). Prefer this over write_file for targeted edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Absolute or relative file path"},
                    "old_str": {"type": "string", "description": "Exact string to find and replace"},
                    "new_str": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files matching a glob pattern. Use '*.ext' for current directory only. Only use '**/*.ext' when you explicitly need to recurse into subdirectories.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py' for current dir or '**/*.py' for recursive"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for a regex pattern in files (like grep -rn).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex to search for"},
                    "path":    {"type": "string", "description": "File or directory to search", "default": "."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a URL and return its content as plain text. Use for documentation, GitHub issues, or any web link the user provides.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to fetch"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via Kagi. Returns ranked results with title, URL, and snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a clarifying question and wait for their answer. Use when you need information you cannot determine from context.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "The question to ask the user"}},
                "required": ["question"],
            },
        },
    },
]


def _progress(pct, stage):
    """Emit a loading progress milestone to stdout (consumed by Go TUI)."""
    print(json.dumps({"loading_pct": pct, "stage": stage}), flush=True)


def load_model(key, quant=None):
    import torch

    _progress(0, "importing transformers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, _, _ = MODELS[key]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_KEY")

    _progress(10, "loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    _progress(25, "loading model weights...")
    if DEVICE == "cuda":
        # CUDA: device_map handles multi-GPU placement; BitsAndBytes quantization available.
        from transformers import BitsAndBytesConfig
        kwargs = {"device_map": "cuda", "token": hf_token}
        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quant == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    else:
        # MPS / CPU: device_map not supported; quantization requires bitsandbytes+CUDA.
        if quant:
            print(f"WARNING: quantization not supported on {DEVICE} — ignoring --quant", file=sys.stderr, flush=True)
        # Use float32 on MPS — torch 2.10+ MPS attention overflows fp16 at
        # longer context lengths (>~1000 tokens), producing NaN. Float32 doubles
        # memory but these models are small enough (<6 GB even at float32).
        dtype = torch.float32 if DEVICE == "mps" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, token=hf_token,
        ).to(DEVICE)

    _progress(85, "model loaded")
    return model, tok


def _prepare_inputs(tok, messages, use_native):
    if use_native:
        text = tok.apply_chat_template(
            messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
        )
    else:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    import torch
    return tok(text, return_tensors="pt").to(DEVICE)


def _has_complete_json(text, start):
    """Check if text[start:] contains a complete JSON object by counting brace depth."""
    depth = 0
    in_string = False
    escape = False
    for ch in text[start:]:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return True
    return False


def _tool_call_stopper(tok, use_native, n_prompt):
    """Return a StoppingCriteria list that halts generation after a complete
    TOOL_CALL JSON line (ReAct format) or </tool_call> tag (native format).
    This prevents small models from hallucinating fake TOOL_RESULT text."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    # Minimum tokens before a tool call could appear
    min_check = 5
    # Check every N tokens to avoid O(n²) decode cost
    check_interval = 4

    class ToolCallStop(StoppingCriteria):
        def __init__(self):
            self.found = False

        def __call__(self, input_ids, scores, **kwargs):
            if self.found:
                return True
            n_gen = input_ids.shape[1] - n_prompt
            if n_gen < min_check:
                return False
            # Only decode every check_interval tokens (and always on min_check)
            if n_gen > min_check and n_gen % check_interval != 0:
                return False
            generated = tok.decode(input_ids[0, n_prompt:], skip_special_tokens=True)
            if use_native:
                # Count <tool_call>...</tool_call> pairs, not substrings
                open_pos = generated.find("<tool_call>")
                if open_pos < 0:
                    return False
                close_pos = generated.find("</tool_call>", open_pos + 11)
                self.found = close_pos >= 0
                return self.found
            # ReAct: find TOOL_CALL: then verify complete JSON via brace depth
            tc_pos = generated.find("TOOL_CALL:")
            if tc_pos < 0:
                return False
            brace_pos = generated.find("{", tc_pos)
            if brace_pos < 0:
                return False
            self.found = _has_complete_json(generated, brace_pos)
            return self.found

    return StoppingCriteriaList([ToolCallStop()])


def generate(model, tok, messages, use_native, max_tokens=1024, temperature=0.2):
    import torch

    inputs = _prepare_inputs(tok, messages, use_native)
    n_in = inputs.input_ids.shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
            stopping_criteria=_tool_call_stopper(tok, use_native, n_in),
        )
    elapsed = time.perf_counter() - t0
    reply = tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
    n_out = out.shape[1] - n_in
    del out, inputs
    _empty_cache()
    return reply, n_out, elapsed


def generate_stream(model, tok, messages, use_native, max_tokens=1024, temperature=0.2):
    """Stream tokens to stdout as JSON lines, then write a done line."""
    import torch, threading
    from transformers import TextIteratorStreamer

    inputs = _prepare_inputs(tok, messages, use_native)
    n_in = inputs.input_ids.shape[1]
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "pad_token_id": tok.eos_token_id,
        "streamer": streamer,
        "stopping_criteria": _tool_call_stopper(tok, use_native, n_in),
    }
    del inputs

    t0 = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    total_tokens = 0
    for token_text in streamer:
        if token_text:
            print(json.dumps({"token": token_text}), flush=True)
        total_tokens += 1

    thread.join()
    elapsed = time.perf_counter() - t0
    del gen_kwargs
    _empty_cache()
    print(json.dumps({"done": True, "tokens": total_tokens, "seconds": round(elapsed, 2)}), flush=True)


def main():
    import warnings
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser(description="LLM sidecar for pai")
    ap.add_argument("--model", "-m", default="qwen-0.5b", choices=list(MODELS))
    ap.add_argument("--quant", "-q", default=None, choices=["4bit", "8bit"],
                    help="Quantization: 4bit (~1.7GB for 3b) or 8bit (~3.5GB for 3b)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    model, tok = load_model(args.model, args.quant)
    import torch
    if DEVICE == "cuda":
        vram_mb = torch.cuda.memory_allocated() // 1024 // 1024
    else:
        # MPS/CPU: estimate from model parameter bytes
        try:
            vram_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) // 1024 // 1024
        except Exception:
            vram_mb = 0
    load_s = time.perf_counter() - t0

    # Probe native tool-calling support
    _progress(90, "probing tool support...")
    use_native = True
    try:
        tok.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tools=TOOLS, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        use_native = False

    # Some models produce NaN with native tool templates on MPS + fp16 (the
    # template inflates prompts past the fp16 precision boundary). The catalog
    # flags these as mps_native_stable=false.
    spec = MODEL_CATALOG.get(args.model, {})
    if use_native and DEVICE == "mps" and not spec.get("mps_native_stable", True):
        use_native = False
        print(f"NOTE: forcing ReAct mode for {args.model} on MPS (fp16 stability)", file=sys.stderr, flush=True)

    # Ready handshake
    print(json.dumps({
        "status":     "ready",
        "model":      args.model,
        "vram_mb":    vram_mb,
        "load_s":     round(load_s, 1),
        "use_native": use_native,
    }), flush=True)

    # Serve requests
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req      = json.loads(raw_line)
            messages = req.get("messages", [])
            max_tok  = req.get("max_tokens", 1024)
            temp     = req.get("temperature", 0.2)
            if req.get("stream"):
                generate_stream(model, tok, messages, use_native, max_tok, temp)
            else:
                reply, tokens, seconds = generate(model, tok, messages, use_native, max_tok, temp)
                print(json.dumps({"reply": reply, "tokens": tokens, "seconds": round(seconds, 2)}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
