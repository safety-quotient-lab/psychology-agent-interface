# psyai

A Claude Code-like agentic TUI backed by a local LLM running on-device via CUDA.

## Architecture

```
psyai (Go TUI)  ←──JSON-lines stdio──→  scripts/llm-serve.py (Python sidecar)
```

The Go binary handles the TUI, tools, parsing, and conversation management.
The Python sidecar loads the model once into VRAM and serves inference requests
for the session lifetime via newline-delimited JSON on stdin/stdout.

## Requirements

- Go 1.21+
- Python 3.10+ with `torch`, `transformers`
- NVIDIA GPU with CUDA 12+ (GTX 1060 6GB minimum for ≤1.5B models)
- HuggingFace token in `~/.dev.vars` as `HUGGINGFACE_KEY=...` (for gated models)

## Models

| Key | Model | VRAM | Notes |
|---|---|---|---|
| `qwen-0.5b` | Qwen2.5-0.5B-Instruct | ~1GB | Fastest |
| `qwen-1.5b` | Qwen2.5-1.5B-Instruct | ~3GB | Default, native tool-calling |
| `qwen-3b` | Qwen2.5-3B-Instruct | ~5.6GB | Needs 6GB+ |
| `smollm2` | SmolLM2-1.7B-Instruct | ~3GB | |
| `gemma-2b` | gemma-2-2b-it | ~4GB | Gated |
| `llama-1b` | Llama-3.2-1B-Instruct | ~2GB | Gated |
| `llama-3b` | Llama-3.2-3B-Instruct | ~6GB | Gated |

## Build & Run

```bash
go build -o psyai .
./psyai --model qwen-1.5b --cwd /path/to/project
```

## Flags

```
--model/-m   Model key (default: qwen-1.5b)
--cwd/-C     Working directory for tools (default: cwd)
--max-turns/-n  Tool-call limit per exchange (default: 15)
--yes/-y     Auto-approve all tool calls
--budget N   Stop after N total tokens (0 = unlimited)
```

## Slash Commands

```
/help          Show this list
/clear         Reset conversation
/cwd <path>    Change working directory
/model <name>  Switch model (restarts sidecar)
/system [text] Show or set system prompt
/export        Save conversation as Markdown to cwd
/session save|list|load <n>
/quit  /exit
```

## Tools

The LLM has access to: `shell`, `read_file`, `write_file`, `list_files`, `search`, `ask_user`.

Shell commands and `write_file` require approval (`y/N`) unless `--yes` is set.
Shell builtins (`cd`, `ls`, `pwd`, `git`, etc.) bypass the LLM entirely via `exec.LookPath`.
