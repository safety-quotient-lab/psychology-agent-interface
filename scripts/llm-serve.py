#!/usr/bin/env python3
"""
llm-serve.py — Persistent LLM sidecar for pai (Go TUI).

Loads a model once, keeps it in VRAM, and serves inference requests
via newline-delimited JSON on stdin/stdout.

Backends:
  - MLX (Apple Silicon): mlx-lm, native Metal acceleration, 4-bit quantization
  - Transformers (Linux/CUDA/CPU): HuggingFace transformers + torch

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
    {
        "type": "function",
        "function": {
            "name": "lsp_definition",
            "description": "Jump to the definition of a Go symbol at a given file position. Returns file:line:col.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Go source file path"},
                    "line": {"type": "integer", "description": "Line number (1-based)"},
                    "col":  {"type": "integer", "description": "Column number (1-based)"},
                },
                "required": ["path", "line", "col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_references",
            "description": "Find all references to a Go symbol at a given file position. Returns file:line:col for each reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Go source file path"},
                    "line": {"type": "integer", "description": "Line number (1-based)"},
                    "col":  {"type": "integer", "description": "Column number (1-based)"},
                },
                "required": ["path", "line", "col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_hover",
            "description": "Show type signature and documentation for a Go symbol at a given file position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Go source file path"},
                    "line": {"type": "integer", "description": "Line number (1-based)"},
                    "col":  {"type": "integer", "description": "Column number (1-based)"},
                },
                "required": ["path", "line", "col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_diagnostics",
            "description": "Get compiler errors and warnings for a Go source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Go source file path"},
                },
                "required": ["path"],
            },
        },
    },
]


def _progress(pct, stage):
    """Emit a loading progress milestone to stdout (consumed by Go TUI)."""
    print(json.dumps({"loading_pct": pct, "stage": stage}), flush=True)


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


def _has_complete_tool_call(text, use_native):
    """Check if generated text contains a complete tool call."""
    if use_native:
        open_pos = text.find("<tool_call>")
        if open_pos < 0:
            return False
        close_pos = text.find("</tool_call>", open_pos + 11)
        return close_pos >= 0
    tc_pos = text.find("TOOL_CALL:")
    if tc_pos < 0:
        return False
    brace_pos = text.find("{", tc_pos)
    if brace_pos < 0:
        return False
    return _has_complete_json(text, brace_pos)


# ---------------------------------------------------------------------------
# Backend: MLX (Apple Silicon)
# ---------------------------------------------------------------------------

class MLXBackend:
    """Inference backend using mlx-lm. Native Metal acceleration on Apple Silicon."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self, key, quant=None):
        _progress(0, "importing mlx-lm...")
        from mlx_lm import load as mlx_load

        model_id, _, _ = MODELS[key]
        # mlx-lm uses huggingface_hub which reads HF_TOKEN from env automatically.
        # Ensure the env var exists (our .dev.vars may set HUGGINGFACE_KEY instead).
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_KEY")
        if hf_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = hf_token

        _progress(10, "loading tokenizer...")
        _progress(25, "loading model weights (MLX)...")

        self.model, self.tokenizer = mlx_load(model_id)
        _progress(85, "model loaded (MLX)")

    def vram_mb(self):
        """Estimate model memory from parameter count (unified memory)."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            total_bytes = sum(
                v.nbytes for _, v in nn.utils.tree_flatten(self.model.parameters())
            )
            return total_bytes // 1024 // 1024
        except Exception:
            return 0

    def probe_native(self, model_key):
        """Check if the tokenizer supports native tool-calling templates."""
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}],
                tools=TOOLS, tokenize=False, add_generation_prompt=True,
            )
            return True
        except Exception:
            return False

    def _build_prompt(self, messages, use_native):
        if use_native:
            return self.tokenizer.apply_chat_template(
                messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
            )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _make_sampler(self, temperature):
        from mlx_lm.sample_utils import make_sampler
        return make_sampler(temp=temperature if temperature > 0 else 0.0, min_p=0.05)

    def generate(self, messages, use_native, max_tokens=1024, temperature=0.2):
        from mlx_lm import stream_generate

        prompt = self._build_prompt(messages, use_native)
        sampler = self._make_sampler(temperature)
        t0 = time.perf_counter()

        # Use streaming internally so we can stop on tool call completion
        full_text = ""
        total_tokens = 0
        for resp in stream_generate(
            self.model, self.tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=sampler,
        ):
            full_text += resp.text
            total_tokens += 1
            if total_tokens >= 5 and _has_complete_tool_call(full_text, use_native):
                break

        elapsed = time.perf_counter() - t0
        return full_text.strip(), total_tokens, elapsed

    def generate_stream(self, messages, use_native, max_tokens=1024, temperature=0.2):
        from mlx_lm import stream_generate

        prompt = self._build_prompt(messages, use_native)
        sampler = self._make_sampler(temperature)
        t0 = time.perf_counter()

        full_text = ""
        total_tokens = 0
        for resp in stream_generate(
            self.model, self.tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=sampler,
        ):
            if resp.text:
                print(json.dumps({"token": resp.text}), flush=True)
            full_text += resp.text
            total_tokens += 1
            if total_tokens >= 5 and _has_complete_tool_call(full_text, use_native):
                break

        elapsed = time.perf_counter() - t0
        print(json.dumps({"done": True, "tokens": total_tokens, "seconds": round(elapsed, 2)}), flush=True)


# ---------------------------------------------------------------------------
# Backend: Transformers (Linux/CUDA/CPU, MPS fallback)
# ---------------------------------------------------------------------------

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


class TransformersBackend:
    """Inference backend using HuggingFace transformers + PyTorch."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = _detect_device()

    def _empty_cache(self):
        try:
            import torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    def load(self, key, quant=None):
        import torch
        _progress(0, "importing transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id, _, _ = MODELS[key]
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_KEY")

        _progress(10, "loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

        _progress(25, "loading model weights...")
        if self.device == "cuda":
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
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        else:
            if quant:
                print(f"WARNING: quantization not supported on {self.device} — ignoring --quant",
                      file=sys.stderr, flush=True)
            # MPS: bfloat16 avoids fp16 NaN overflow, same memory as fp16, ~2x vs float32.
            if self.device == "mps":
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, token=hf_token,
            ).to(self.device)

        _progress(85, "model loaded")

    def vram_mb(self):
        import torch
        if self.device == "cuda":
            return torch.cuda.memory_allocated() // 1024 // 1024
        try:
            return sum(p.nelement() * p.element_size() for p in self.model.parameters()) // 1024 // 1024
        except Exception:
            return 0

    def probe_native(self, model_key):
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}],
                tools=TOOLS, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return False
        # Some models produce NaN with native tool templates on MPS.
        spec = MODEL_CATALOG.get(model_key, {})
        if self.device == "mps" and not spec.get("mps_native_stable", True):
            print(f"NOTE: forcing ReAct mode for {model_key} on MPS (stability)",
                  file=sys.stderr, flush=True)
            return False
        return True

    def _prepare_inputs(self, messages, use_native):
        import torch
        if use_native:
            text = self.tokenizer.apply_chat_template(
                messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
            )
        else:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def _tool_call_stopper(self, use_native, n_prompt):
        from transformers import StoppingCriteria, StoppingCriteriaList

        min_check = 5
        check_interval = 4
        tok = self.tokenizer

        class ToolCallStop(StoppingCriteria):
            def __init__(self):
                self.found = False

            def __call__(self, input_ids, scores, **kwargs):
                if self.found:
                    return True
                n_gen = input_ids.shape[1] - n_prompt
                if n_gen < min_check:
                    return False
                if n_gen > min_check and n_gen % check_interval != 0:
                    return False
                generated = tok.decode(input_ids[0, n_prompt:], skip_special_tokens=True)
                self.found = _has_complete_tool_call(generated, use_native)
                return self.found

        return StoppingCriteriaList([ToolCallStop()])

    def generate(self, messages, use_native, max_tokens=1024, temperature=0.2):
        import torch

        inputs = self._prepare_inputs(messages, use_native)
        n_in = inputs.input_ids.shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self._tool_call_stopper(use_native, n_in),
            )
        elapsed = time.perf_counter() - t0
        reply = self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        n_out = out.shape[1] - n_in
        del out, inputs
        self._empty_cache()
        return reply, n_out, elapsed

    def generate_stream(self, messages, use_native, max_tokens=1024, temperature=0.2):
        import torch, threading
        from transformers import TextIteratorStreamer

        inputs = self._prepare_inputs(messages, use_native)
        n_in = inputs.input_ids.shape[1]
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "repetition_penalty": 1.15,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
            "stopping_criteria": self._tool_call_stopper(use_native, n_in),
        }
        del inputs

        t0 = time.perf_counter()
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        total_tokens = 0
        for token_text in streamer:
            if token_text:
                print(json.dumps({"token": token_text}), flush=True)
            total_tokens += 1

        thread.join()
        elapsed = time.perf_counter() - t0
        del gen_kwargs
        self._empty_cache()
        print(json.dumps({"done": True, "tokens": total_tokens, "seconds": round(elapsed, 2)}), flush=True)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _select_backend():
    """Pick MLX on Apple Silicon when available, otherwise Transformers."""
    try:
        import platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            import mlx_lm  # noqa: F401
            print("NOTE: using MLX backend (Apple Silicon)", file=sys.stderr, flush=True)
            return MLXBackend()
    except ImportError:
        print("NOTE: mlx-lm not installed — falling back to transformers", file=sys.stderr, flush=True)
    except Exception:
        pass
    return TransformersBackend()


# ---------------------------------------------------------------------------
# Main loop (backend-agnostic)
# ---------------------------------------------------------------------------

def main():
    import warnings
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser(description="LLM sidecar for pai")
    ap.add_argument("--model", "-m", default="qwen-0.5b", choices=list(MODELS))
    ap.add_argument("--quant", "-q", default=None, choices=["4bit", "8bit"],
                    help="Quantization: 4bit or 8bit (CUDA only for transformers, native for MLX)")
    args = ap.parse_args()

    backend = _select_backend()
    backend_name = type(backend).__name__

    t0 = time.perf_counter()
    backend.load(args.model, args.quant)
    vram_mb = backend.vram_mb()
    load_s = time.perf_counter() - t0

    # Probe native tool-calling support
    _progress(90, "probing tool support...")
    use_native = backend.probe_native(args.model)

    # Ready handshake
    print(json.dumps({
        "status":     "ready",
        "model":      args.model,
        "vram_mb":    vram_mb,
        "load_s":     round(load_s, 1),
        "use_native": use_native,
        "backend":    backend_name,
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
                backend.generate_stream(messages, use_native, max_tok, temp)
            else:
                reply, tokens, seconds = backend.generate(messages, use_native, max_tok, temp)
                print(json.dumps({"reply": reply, "tokens": tokens, "seconds": round(seconds, 2)}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
