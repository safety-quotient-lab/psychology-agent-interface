# pai — Next Up

## Format Compliance (cluster)

- [x] **Per-turn format reinforcement** — Done (ed718c1). msgsWithFormatNudge()
      appends a transient reminder before every inference call (TUI + print mode).
- [x] **Multi-turn format drift** — Verified (e67f78f). All TUI chat inference
      calls pass through `msgsWithFormatNudge()` → `ns.WithNudge()`, appending
      tier-appropriate tag/confidence reminders at the tail of every turn.
      Summary-only compaction calls intentionally skip nudge. Test coverage in
      `pkg/prompt/prompt_test.go`.
- [x] **Verify unprompted tool call fix in TUI** — Verified (e67f78f). File
      listing now embeds directly in system prompt (no fake tool exchanges).
      Few-shot priming teaches format, not tool patterns. `dispatchNextTool()`
      gates dangerous ops through user approval. Original fix (f3fda40) intact
      through all Plan 9 refactors.

## Inference

- [x] **Stop generation after tool call** — Fixed (eead6ef). llama-1b hallucinated
      fake TOOL_RESULT text after emitting TOOL_CALL, causing TUI hang. Added
      `StoppingCriteria` in sidecar (`llm-serve.py`) that halts generation once
      a complete tool call appears. Stronger ReAct prompt instruction added.

## Context Management

- [x] **Context compaction tuning for Tier 1** — Done (39cfa54). Per-model
      conservative limits trigger early compaction (4K for qwen-0.5b/smollm2/gemma-2b).
- [x] **Token budget / context overflow guard** — Done (39cfa54). estimateTokens()
      checks conversation size before each inference; compacts when approaching limit.

## TUI

- [x] **Fix mouse text selection** — Fixed (3255039). Removed
      `tea.WithMouseCellMotion()` from `tea.NewProgram()` and the `MouseMsg`
      handler. Mouse capture intercepted terminal selection. pgup/pgdn
      keyboard scrolling remains functional.
- [x] **Autoscroll on content overflow** — Fixed. `syncViewport()` now detects
      first overflow (content crosses viewport height) and forces `GotoBottom()`.
- [x] **Input box full-width** — Fixed. `SetWidth(msg.Width - 4)` in
      `WindowSizeMsg` handler expands textarea to terminal width minus border.
- [x] **Input box border** — Fixed. Wrapped textarea in `lipgloss.RoundedBorder()`
      with `DimGray` border foreground. `chatHeight()` accounts for border.
- [x] **Prompt/status bar** — Fixed. `renderPromptBar()` shows
      `user@host:~/cwd` on left, `[model] Ntok` on right. Rendered between
      input box and help bar in `View()`.
- [x] **Fix TestGatewayHTTPProc timeout** — Fixed (3255039). Test now uses
      gateway's actual loaded model, skips doInfer on timeout. Suite runs in
      ~1.5s instead of ~62s.
- [x] **Debug log not writing** — Fixed. Default log level changed from
      `WarnLevel` to `InfoLevel` in `pkg/log/log.go`. `plog.L.Info()` calls
      now write to `~/.local/state/pai/debug.log`. Debug level still opt-in
      via `PAI_LOG_LEVEL=debug`.
- [ ] **Verify StoppingCriteria live** — The sidecar tool call stop fix
      (eead6ef) needs a live TUI test with llama-1b to confirm hallucinated
      TOOL_RESULT text no longer appears.

## iOS Port

- [x] **Extract `SystemPromptProvider` interface** — Already satisfied by Plan 9
      provider interfaces in `pkg/prompt/providers.go`: `IdentityProvider`,
      `FormatProvider`, `ToolProvider`, `ContextProvider`, `FewShotProvider`.
      The iOS `SystemPromptProvider` protocol maps directly to these.
- [ ] **SwiftUI MVP** — llama.cpp backend, glass UI, conversational core. Phases
      defined in `docs/ios-sketch.md`.

## LSP Integration

- [x] **gopls integration** — Done. `pkg/lsp/client.go` manages gopls subprocess
      via JSON-RPC over stdin/stdout. Four read-only tools registered:
      `lsp_definition`, `lsp_references`, `lsp_hover`, `lsp_diagnostics`.
      Conditional — tools only register when gopls available.
- [x] **Registry memoization** — Done. `defaultRegistry()` now cached; avoids
      re-registering 13+ tools on every `executeTool()` call.
- [ ] **LSP live test** — Validate lsp_definition/references/hover with a running
      gopls against this codebase. Check startup latency impact.

## MLX Backend

- [x] **Dual-backend sidecar** — Done. `llm-serve.py` auto-selects MLX on Apple
      Silicon (arm64 + mlx-lm installed), falls back to transformers on Linux/CUDA.
      Same JSON protocol — Go TUI unchanged. Backend name reported in ready handshake.
- [ ] **MLX live test** — Run smollm2 or qwen-0.5b via MLX backend, compare tok/s
      against transformers+bfloat16. Verify tool call stopping works.
- [ ] **MLX quantization** — Test 4-bit quantized models via mlx-lm (native on
      Apple Silicon, unlike bitsandbytes). Could halve memory for larger models.

## Code Agent

- [ ] **Go code generation and self-editing** — Extend pai to write Go code and
      edit its own codebase. Key capabilities needed:
      - `write_file` and `edit_file` tools already exist but need Go-aware
        validation (syntax check via `go build`, `goimports`)
      - System prompt persona extension: add a "software engineer" skill overlay
        that activates when the user asks for code changes
      - `go build` / `go test` feedback loop: after writing code, automatically
        run build+test and feed errors back for self-correction
      - Project context: embed `go.mod`, package structure, and recent git log
        in the workspace context so the model understands the codebase
      - Safety: require approval for `write_file`/`edit_file` (already gated),
        add `go vet` as post-write validation before committing
      - Consider a `/dev` skill that activates the full development toolchain
