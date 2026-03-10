# pai — Next Up

## Format Compliance (cluster)

- [ ] **Per-turn format reinforcement** — Inject a compact format reminder into
      the conversation before each inference call, not just once in the system
      prompt. Small models lose tag compliance after the first exchange when the
      system prompt drifts out of their attention window.
- [ ] **Multi-turn format drift** — Verify that per-turn reinforcement resolves
      the tag dropout observed in TUI multi-turn sessions (replays show
      `[observation]`/`[inference]` tags missing from turn 2 onward).
- [ ] **Verify unprompted tool call fix in TUI** — The fake tool-exchange removal
      (commit f3fda40) only tested in print mode. Confirm the model no longer
      fires unsolicited `list_files` calls in interactive TUI sessions.

## Context Management

- [ ] **Context compaction tuning for Tier 1** — Small models (<=2B) degrade past
      ~2-4K tokens. Tune the compaction/summary trigger threshold so it fires
      earlier for Tier 1, preserving coherence across longer sessions.
- [ ] **Token budget / context overflow guard** — smollm2 and gemma-2b have 8K
      context limits. The sidecar does not cap input context. Add a guard that
      truncates or compacts before exceeding the model's declared max context.

## iOS Port

- [ ] **Extract `SystemPromptProvider` interface** — Decouple prompt content from
      inference/UI in the Go codebase. Required before the Swift port can share
      prompt logic. Design sketch at `docs/ios-sketch.md`.
- [ ] **SwiftUI MVP** — llama.cpp backend, glass UI, conversational core. Phases
      defined in `docs/ios-sketch.md`.
