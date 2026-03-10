# pai — Next Up

## Format Compliance (cluster)

- [x] **Per-turn format reinforcement** — Done (ed718c1). msgsWithFormatNudge()
      appends a transient reminder before every inference call (TUI + print mode).
- [ ] **Multi-turn format drift** — Verify in TUI that per-turn nudge resolves
      tag dropout on turn 2+. Run a multi-turn session and check replay.
- [ ] **Verify unprompted tool call fix in TUI** — Code verified: no fake tool
      exchanges remain. Needs TUI session replay to confirm end-to-end.

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

## iOS Port

- [ ] **Extract `SystemPromptProvider` interface** — Decouple prompt content from
      inference/UI in the Go codebase. Required before the Swift port can share
      prompt logic. Design sketch at `docs/ios-sketch.md`.
- [ ] **SwiftUI MVP** — llama.cpp backend, glass UI, conversational core. Phases
      defined in `docs/ios-sketch.md`.
