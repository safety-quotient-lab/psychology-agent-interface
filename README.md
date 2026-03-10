# psyai

Socratic psychology agent interface — local LLM-powered structured inquiry.

Forked from [local-cc](https://github.com/kashfshah/local-cc). Uses tiered
psychology-agent system prompts from
[safety-quotient-lab/psychology-agent](https://github.com/safety-quotient-lab/psychology-agent)
distilled for small local models.

## Architecture

```
psyai (Go TUI)  ←──JSON-lines stdio──→  scripts/llm-serve.py (Python sidecar)
```

The Go binary handles the TUI, tools, parsing, and conversation management.
The Python sidecar loads the model once into VRAM/MPS and serves inference
requests for the session lifetime via newline-delimited JSON on stdin/stdout.

## Psychology Agent Prompt Tiers

System prompts adapt automatically based on model parameter count:

| Tier | Params | Models | Capabilities |
|------|--------|--------|-------------|
| 1 | ≤2B | qwen-0.5b, qwen-1.5b, llama-1b, smollm2 | Role framing, [observation]/[inference] tags, confidence footer, hard refusals |
| 2 | 2B–4B | qwen-3b, llama-3b, gemma-2b | + [OBS]/[INF] tags, evidence linking, uncertainty dimensions, anti-sycophancy |
| 3 | 4B–8B | (future models) | + Socratic stance, interpretant awareness, recommend-against, scope boundaries |

At Q4 quantization, drop one tier (e.g., a 3B Q4 model gets the ≤2B prompt).

## Socratic Method

The core interaction pattern uses Socratic questioning:

- **Critic loop**: after each response, optionally generates 1-2 follow-up
  questions that surface unstated assumptions and invite deeper examination
- **Socratic inquiry skill**: structured questioning protocol (assumption,
  evidence, alternative, implication, definition, perspective)
- **Differential diagnosis skill**: hypothesis scoring on 5 dimensions for
  structured evaluation

## Requirements

- Go 1.21+
- Python 3.10+ with `torch`, `transformers`
- GPU with CUDA 12+ or Apple Silicon (MPS), or CPU fallback
- HuggingFace token in `~/.dev.vars` as `HUGGINGFACE_KEY=...` (for gated models)

## Models

| Key | Model | VRAM | Notes |
|---|---|---|---|
| `qwen-0.5b` | Qwen2.5-0.5B-Instruct | ~1GB | Fastest, tier 1 |
| `qwen-1.5b` | Qwen2.5-1.5B-Instruct | ~3GB | Default, native tool-calling, tier 1 |
| `qwen-3b` | Qwen2.5-3B-Instruct | ~5.6GB | Tier 2, richer psychology prompts |
| `smollm2` | SmolLM2-1.7B-Instruct | ~3GB | Tier 1 |
| `gemma-2b` | gemma-2-2b-it | ~4GB | Gated, tier 2 |
| `llama-1b` | Llama-3.2-1B-Instruct | ~2GB | Gated, tier 1 |
| `llama-3b` | Llama-3.2-3B-Instruct | ~6GB | Gated, tier 2 |

## Build & Run

```bash
go build -o psyai .
./psyai --model qwen-3b --cwd /path/to/project
```

## Flags

```
--model/-m      Model key (default: qwen-3b)
--cwd/-C        Working directory for tools (default: cwd)
--max-turns/-n  Tool-call limit per exchange (default: 15)
--yes/-y        Auto-approve all tool calls
--no-review     Disable Socratic follow-up questions
--budget N      Stop after N total tokens (0 = unlimited)
```

## Skills

```
/skill socratic-inquiry    Structured Socratic questioning on a topic or claim
/skill differential-diagnosis  Hypothesis scoring (5 dimensions, threshold ≥20/25)
/skill off                 Deactivate current skill
```

## Tools

The LLM has access to: `shell`, `read_file`, `write_file`, `edit_file`,
`list_files`, `search`, `fetch_url`, `web_search`, `ask_user`.

Shell commands and file writes require approval (`y/N`) unless `--yes` is set.

## Provenance

Psychology agent system prompts distilled from
[safety-quotient-lab/psychology-agent](https://github.com/safety-quotient-lab/psychology-agent)
cogarch (Session 48, 2026-03-09). See `docs/lite-system-prompt.md` in that repo
for design rationale and testing protocol.
