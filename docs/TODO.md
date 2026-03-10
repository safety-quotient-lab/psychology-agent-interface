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

- [ ] **Fix mouse text selection** — Users cannot select text in the TUI for
      copy/paste. Likely a bubbletea mouse event capture conflict. Investigate
      whether disabling mouse reporting or switching to a zone-based approach
      restores native terminal selection.
- [ ] **Autoscroll on content overflow** — First content overflow requires manual
      scroll. `syncViewport()` calls `GotoBottom()` only when `wasAtBottom ||
      activeStream` — check if `wasAtBottom` returns false on the initial
      overflow transition. May need to force-scroll when content first exceeds
      viewport height.
- [ ] **Input box full-width** — textarea does not expand to terminal width.
      `newTextarea()` at model.go:243 sets no explicit width. Add
      `ta.SetWidth(msg.Width)` in `WindowSizeMsg` handler and initial setup.
- [ ] **Input box border** — Add a lipgloss border style to the textarea area
      in `View()`. Use `lipgloss.RoundedBorder()` consistent with tool result
      boxes. Account for border in `chatHeight()` chrome calculation.
- [ ] **Prompt/status bar below input** — Show username, machine name, cwd,
      model, and session stats below the input box. Use `os/user.Current()` and
      `os.Hostname()`. Render as a styled line between input and help bar in
      `View()`. Update `chatHeight()` chrome to account for the extra line.
- [ ] **Fix TestGatewayHTTPProc timeout** — Test connects to port 7705 with no
      server, always times out after 60s. Either add a test fixture/mock or
      `t.Skip()` when gateway unavailable.
- [ ] **Debug log not writing** — `~/.local/state/pai/debug.log` stays at 0
      bytes despite `plog.L.Debug()` calls throughout model.go. Check logger
      initialization and default log level.
- [ ] **Verify StoppingCriteria live** — The sidecar tool call stop fix
      (eead6ef) needs a live TUI test with llama-1b to confirm hallucinated
      TOOL_RESULT text no longer appears.

## iOS Port

- [ ] **Extract `SystemPromptProvider` interface** — Decouple prompt content from
      inference/UI in the Go codebase. Required before the Swift port can share
      prompt logic. Design sketch at `docs/ios-sketch.md`.
- [ ] **SwiftUI MVP** — llama.cpp backend, glass UI, conversational core. Phases
      defined in `docs/ios-sketch.md`.
