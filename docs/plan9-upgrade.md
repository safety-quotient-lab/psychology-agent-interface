# pai — Plan 9 Upgrade

Status: **complete** — all 6 phases implemented.

---

## Design Thesis

Plan 9 reduces system complexity through three interlocking ideas:

1. **Resources as files** — every service presents a file interface
2. **Per-process namespaces** — each process composes its own view of resources
3. **Plumbing** — messages route to handlers by pattern, not hardcoded dispatch

pai currently hardcodes tools in a switch statement, tangles prompt construction
with model detection, scatters configuration across env vars and struct fields,
and couples the TUI state machine to inference details. A Plan 9-style refactor
decomposes these into composable, data-driven pieces that the iOS port inherits
directly.

---

## Phase 1 — Tool Registry (resources as files)

### Problem

`executeTool()` in tools.go contains a monolithic switch over 9 tool names. Adding
a tool means editing Go code, editing Python tool definitions in llm-serve.py,
and updating system prompt strings in model.go. Three files stay synchronized by
convention — nothing enforces consistency.

### Plan 9 principle

Expose tools as a registry loaded from data, not compiled into a dispatcher. Each
tool becomes a self-describing entry (name, description, parameters, executor)
that the system discovers at startup.

### Design

```go
// pkg/tool/tool.go

// Tool describes a callable tool. Mirrors Plan 9's file interface:
// the name identifies it, the schema describes how to "write" to it,
// and Execute "reads" the result.
type Tool struct {
    Name        string              `json:"name"`
    Description string              `json:"description"`
    Parameters  []Param             `json:"parameters"`
    Execute     func(args map[string]any, cwd string) string `json:"-"`
}

type Param struct {
    Name        string `json:"name"`
    Type        string `json:"type"`
    Description string `json:"description"`
    Required    bool   `json:"required"`
    Default     string `json:"default,omitempty"`
}

// Registry holds available tools, keyed by name.
type Registry struct {
    tools map[string]Tool
}

func NewRegistry() *Registry { return &Registry{tools: make(map[string]Tool)} }

func (r *Registry) Register(t Tool)                          { r.tools[t.Name] = t }
func (r *Registry) Get(name string) (Tool, bool)             { return r.tools[name] }
func (r *Registry) Execute(name string, args map[string]any, cwd string) string { ... }
func (r *Registry) Names() []string                          { ... } // sorted
func (r *Registry) Descriptions() []ToolDescription          { ... } // for prompt building
func (r *Registry) JSONSchemas() []map[string]any            { ... } // for native tool calling
```

### Changes

| File | Action |
|------|--------|
| `pkg/tool/tool.go` | **New** — Tool, Param, Registry types |
| `pkg/tool/builtins.go` | **New** — register shell, read_file, write_file, edit_file, list_files, search, fetch_url, web_search, ask_user |
| `tools.go` | **Delete** — all tool logic moves to pkg/tool |
| `model.go` | Replace `executeTool(name, args, cwd)` calls with `registry.Execute(name, args, cwd)` |
| `print.go` | Same replacement |
| `scripts/llm-serve.py` | Generate TOOLS list from registry export (or read from shared JSON) |

### Shared tool definitions

Write a canonical `tools.json` that both Go and Python read:

```
scripts/tools.json    — single source of truth for tool schemas
```

Go loads it into the Registry at startup. Python loads it in llm-serve.py.
`nativeSystem()` and `reactSystem()` generate tool descriptions from the registry
rather than hardcoded strings.

### Benefits

- Adding a tool means adding one function + one JSON entry. No switch editing.
- The iOS port reads the same `tools.json` and maps executors per platform.
- `PlatformCapabilities` (from ios-sketch.md) filters the registry at startup:
  iOS removes `shell`, sandboxes file tools.

---

## Phase 2 — Prompt Namespace (per-process namespaces)

### Problem

System prompt construction scatters across `psychologyPromptTier()`,
`nativeSystem()`, `reactSystem()`, `fewShotPriming()`, `formatReminder()`,
`msgsWithFormatNudge()`, and `criticPrompt()`. Each function reaches into global
state (model name, env vars, file lists) to assemble its piece. The iOS port
cannot reuse this — the logic sits in Go functions that reference Go-only
dependencies.

### Plan 9 principle

Each session composes its own namespace from independent resources. A prompt
namespace contains: identity (who), tools (what), context (where), format rules
(how), and few-shot examples (show). Each resource provides its content
independently; the namespace assembles them.

### Design

```go
// pkg/prompt/prompt.go

// Namespace assembles a complete system prompt from independent resources.
// Each resource provides content without knowing about the others.
type Namespace struct {
    Identity   IdentityProvider
    Tools      ToolProvider
    Context    ContextProvider
    Format     FormatProvider
    FewShot    FewShotProvider
}

// Build composes the final system prompt and priming messages.
func (ns Namespace) Build() (systemPrompt string, priming []Message) { ... }

// NudgeMessage returns the per-turn format reinforcement message.
func (ns Namespace) NudgeMessage() Message { ... }
```

### Provider interfaces

```go
// IdentityProvider returns the persona prompt for a given model tier.
type IdentityProvider interface {
    Name() string
    Prompt(tier int) string
}

// ToolProvider returns tool descriptions for the prompt.
// Filtered by platform capabilities at construction time.
type ToolProvider interface {
    ToolList() string         // "shell, read_file, ..." for system prompt
    ToolDescriptions() string // full descriptions for ReAct mode
}

// ContextProvider returns workspace context (file list, project instructions).
type ContextProvider interface {
    WorkingDir() string
    FileList() string
    ProjectInstructions() string
}

// FormatProvider returns format rules and reminders for a model tier.
type FormatProvider interface {
    Rules(tier int) string    // appended to system prompt
    Nudge(tier int) string    // per-turn reinforcement
}

// FewShotProvider returns example exchanges for a model tier.
type FewShotProvider interface {
    Examples(tier int) []Message
}
```

### Concrete implementations

```go
// pkg/prompt/psychology.go
type PsychologyIdentity struct{}           // implements IdentityProvider
type PsychologyFormat struct{}             // implements FormatProvider
type PsychologyFewShot struct{}            // implements FewShotProvider

// pkg/prompt/workspace.go
type WorkspaceContext struct {              // implements ContextProvider
    CWD      string
    ClaudeMD string
    Registry *tool.Registry
}

// pkg/prompt/toolprompt.go
type RegistryToolProvider struct {          // implements ToolProvider
    Registry *tool.Registry
    Native   bool
}
```

### Changes

| File | Action |
|------|--------|
| `pkg/prompt/prompt.go` | **New** — Namespace, Build, NudgeMessage |
| `pkg/prompt/providers.go` | **New** — provider interfaces |
| `pkg/prompt/psychology.go` | **New** — PsychologyIdentity/Format/FewShot (moves from model.go) |
| `pkg/prompt/workspace.go` | **New** — WorkspaceContext (moves from model.go) |
| `pkg/prompt/toolprompt.go` | **New** — RegistryToolProvider |
| `model.go` | Remove `psychologyPromptTier()`, `nativeSystem()`, `reactSystem()`, `fewShotPriming()`, `formatReminder()`, `msgsWithFormatNudge()`. Replace with `ns.Build()` / `ns.NudgeMessage()` |

### Benefits

- iOS port implements the same provider interfaces in Swift. The Namespace
  assembly logic translates line-for-line.
- Adding a new persona (research methods assistant, generic assistant) means
  implementing IdentityProvider + FormatProvider + FewShotProvider. No changes
  to inference, TUI, or tool code.
- Each provider unit-tests independently.

---

## Phase 3 — Message Plumbing (plumber-style dispatch)

### Problem

Tool call parsing and routing spreads across model.go (TUI streaming handler),
print.go (blocking loop), and parser.go (regex extraction). The flow:
parse reply → extract calls → dispatch tool → format result → append message.
This pipeline repeats in three places (TUI streaming, TUI blocking, print mode)
with slight variations.

### Plan 9 principle

The plumber routes messages to handlers by pattern. A tool call message arrives,
gets pattern-matched to a handler, and the result message routes back. The
pipeline becomes a single reusable loop.

### Design

```go
// pkg/agent/loop.go

// Turn represents one inference + tool-execution cycle.
type Turn struct {
    Registry    *tool.Registry
    Namespace   *prompt.Namespace
    ParseReply  func(reply string) []tool.Call  // parseNative or parseReact
    FormatResult func(call tool.Call, result string) Message
}

// RunTool executes one tool call and returns the result message.
func (t Turn) RunTool(call tool.Call, cwd string) Message { ... }

// ProcessReply parses a reply, executes tools, returns result messages.
// Returns nil if the reply contains no tool calls (final answer).
func (t Turn) ProcessReply(reply string, cwd string) []Message { ... }
```

### Blocking loop (print mode)

```go
func runPrint(c appConfig, proc inferProc, turn agent.Turn) error {
    // ... setup ...
    for i := 0; i < maxTurns; i++ {
        ir, err := proc.infer(ns.WithNudge(msgs), 1024, 0.2)
        resultMsgs := turn.ProcessReply(ir.Reply, c.cwd)
        if resultMsgs == nil {
            fmt.Println(stripMarkup(ir.Reply))
            return nil
        }
        msgs = append(msgs, Message{Role: "assistant", Content: ir.Reply})
        msgs = append(msgs, resultMsgs...)
    }
}
```

### TUI streaming (model.go)

The TUI continues using tea.Cmd for async dispatch, but tool execution
routes through the same `Turn.RunTool()`:

```go
case msgToolDone:
    // Already handled — Turn.RunTool called in the tea.Cmd
```

### Changes

| File | Action |
|------|--------|
| `pkg/agent/loop.go` | **New** — Turn, RunTool, ProcessReply |
| `print.go` | Simplify to use Turn.ProcessReply |
| `model.go` | Tool dispatch uses Turn.RunTool via tea.Cmd |
| `parser.go` | Stays — parseNative/parseReact remain, wired into Turn.ParseReply |

### Benefits

- Single tool dispatch pipeline for all modes (TUI, print, future iOS).
- Adding a new execution mode (HTTP API server, batch mode) reuses the same Turn.
- Tool result formatting (native vs ReAct message shape) consolidated in one place.

---

## Phase 4 — Session as Filesystem (text as universal interface)

### Problem

Sessions save to `~/.local/share/pai/sessions/` as JSON blobs and to cwd as
JSONL. Export formats (JSONL, HTML, Markdown) live in session.go as separate
functions. The session data model (Message slice) lacks structure — no metadata
about which tool calls belong to which turn, no prompt namespace snapshot.

### Plan 9 principle

Sessions map to a directory structure — each session becomes a directory
containing typed files:

```
~/.local/share/pai/sessions/
  20260310-120000/
    meta.json         — model, timestamp, namespace config
    conversation.jsonl — messages, one per line
    tools.json        — tool registry snapshot (what tools existed)
    prompt.txt        — assembled system prompt
```

### Design

```go
// pkg/session/session.go

type Session struct {
    Dir      string
    Meta     Meta
    Messages []Message
}

type Meta struct {
    ID        string    `json:"id"`
    Model     string    `json:"model"`
    CreatedAt time.Time `json:"created_at"`
    Persona   string    `json:"persona"`  // identity provider name
    CWD       string    `json:"cwd"`
}

func Create(model, persona, cwd string) (*Session, error) { ... }
func (s *Session) Append(msg Message) error               { ... } // appends to conversation.jsonl
func (s *Session) Export(format string) (string, error)    { ... } // "html", "markdown", "jsonl"
func List() ([]Meta, error)                                { ... }
func Load(id string) (*Session, error)                     { ... }
```

### Changes

| File | Action |
|------|--------|
| `pkg/session/session.go` | **New** — Session, Meta, directory-based storage |
| `pkg/session/export.go` | **New** — JSONL, HTML, Markdown export (moves from session.go) |
| `session.go` | **Delete** — all logic moves to pkg/session |
| `model.go` | Session operations use pkg/session |

### Benefits

- Sessions become inspectable with standard tools (`cat`, `ls`, `jq`).
- Incremental writes (append-only JSONL) prevent data loss on crash.
- The prompt snapshot enables exact session replay with the original system prompt.
- iOS port reads/writes the same directory format via iCloud sync.

---

## Phase 5 — Model Capabilities as Data (namespace composition)

### Problem

Model metadata scatters across: `selectableModels` (display info), `MODELS` dict
in llm-serve.py (HuggingFace IDs), `modelContextLimit` map (token limits),
`isTier1()`/`isTier2()` functions (tier detection), the `fewShotPriming()` switch
(tier → examples), and the `formatReminder()` switch (tier → nudge text). Adding
a model means editing 6+ locations.

### Plan 9 principle

Model capabilities form a namespace entry — a single data description that the
system reads to configure itself. No hardcoded switches.

### Design

```go
// pkg/model/catalog.go

type ModelSpec struct {
    Key          string `json:"key"`           // "qwen-0.5b"
    Label        string `json:"label"`         // "Qwen 2.5 0.5B"
    HuggingFace  string `json:"huggingface"`   // "Qwen/Qwen2.5-0.5B-Instruct"
    Tier         int    `json:"tier"`           // 1, 2, or 3
    ContextLimit int    `json:"context_limit"`  // conservative effective limit
    DeclaredCtx  string `json:"declared_ctx"`   // "32K" for display
    VRAM         string `json:"vram"`           // "~1 GB"
    Dtype        string `json:"dtype"`          // "fp16"
    Gated        bool   `json:"gated"`          // requires HF token
    Notes        string `json:"notes"`          // "fastest", "heavy", etc.
    NativeTools  bool   `json:"native_tools"`   // supports <tool_call> format
    MPSStable    bool   `json:"mps_stable"`     // false for llama on MPS w/ native tools
}

type Catalog struct {
    Models []ModelSpec
}

func LoadCatalog() Catalog                      { ... } // from embedded JSON or file
func (c Catalog) Get(key string) (ModelSpec, bool) { ... }
func (c Catalog) Tier(key string) int           { ... }
func (c Catalog) ContextLimit(key string) int   { ... }
func (c Catalog) Selectable() []ModelSpec       { ... }
```

### Canonical data file

```
models.json — single source of truth for all model metadata
```

Both Go and Python read from this file. The sidecar uses it for HuggingFace IDs,
dtype, and MPS stability flags. Go uses it for the selector UI, tier detection,
context limits, and native tool support decisions.

### Changes

| File | Action |
|------|--------|
| `models.json` | **New** — canonical model catalog |
| `pkg/model/catalog.go` | **New** — ModelSpec, Catalog, LoadCatalog |
| `model.go` | Remove `selectableModels`, `modelContextLimit`, `isTier1()`, `isTier2()`. Replace with Catalog lookups. |
| `scripts/llm-serve.py` | Remove `MODELS` dict. Load from `models.json`. MPS stability check uses catalog data instead of hardcoded model name check. |

### Benefits

- Adding a model means adding one JSON entry. Everything else derives from it.
- The iOS port reads the same catalog to populate ModelPickerView.
- Tier detection, context limits, and native tool support all come from data,
  not scattered if/switch statements.
- The MPS stability fix for Llama (Phase 0) becomes a data flag (`mps_stable: false`)
  rather than a hardcoded model name check.

---

## Phase 6 — Inference as a Service Interface (small programs)

### Problem

`inferProc` already abstracts over llmProc and httpProc, which follows the Plan 9
ethos well. But the interface mixes concerns: inference (`infer`, `startInfer`,
`nextToken`), lifecycle (`waitReady`, `restart`), and critic review
(`startReview`, `nextReviewToken`). The review methods duplicate the streaming
pattern for a different use case.

### Plan 9 principle

Small programs doing one thing. The inference service handles inference. The
critic builds on top of it as a separate concern, not baked into the interface.

### Design

```go
// Simplified inferProc — inference only
type InferProc interface {
    WaitReady() (ReadyPayload, error)
    Infer(msgs []Message, maxTokens int, temp float64) (InferResponse, error)
    StartStream(msgs []Message, maxTokens int, temp float64) tea.Cmd
    NextToken() tea.Cmd
    Restart(model, projectRoot string) (InferProc, error)
}

// Critic composes InferProc — not embedded in the interface
type Critic struct {
    proc InferProc
}
func (c Critic) Start(userMsg, reply string) tea.Cmd { ... }
func (c Critic) NextToken() tea.Cmd                  { ... }
```

### Changes

| File | Action |
|------|--------|
| `main.go` | Remove `startReview`/`nextReviewToken` from inferProc. Create Critic wrapper. |
| `model.go` | Review state uses Critic instead of proc.startReview/nextReviewToken |

### Benefits

- inferProc stays focused on inference.
- The critic becomes a composable layer — can add other post-processing passes
  (fact-check, tone review) without modifying the interface.

---

## Implementation Order

```
Phase 0 ✓  Upgrade transformers + torch; MPS stability fix for Llama
Phase 5 ✓  Model Capabilities as Data     → models.json + pkg/catalog
Phase 1 ✓  Tool Registry                  → pkg/tool (Registry, builtins)
Phase 2 ✓  Prompt Namespace               → pkg/prompt (providers, psychology)
Phase 3 ✓  Message Plumbing               → turn.go (Turn.ProcessReply)
Phase 6 ✓  Inference Interface Cleanup    → reviewer interface extracted
Phase 4 ✓  Session as Filesystem          → pkg/session (save, export)
```

Phases 1 and 5 should come first — they create the shared data files
(`tools.json`, `models.json`) that the iOS port needs. Phase 2 directly
enables the `SystemPromptProvider` extraction from the iOS sketch.
Phase 3 reduces code duplication. Phases 4 and 6 improve the architecture
but carry less urgency.

### Recommended sequence

```
5 → 1 → 2 → 3 → 6 → 4
```

- **5 first**: model catalog eliminates the most scattered duplication (6+ locations)
  and unblocks the sidecar from reading shared data.
- **1 next**: tool registry removes the monolithic switch and creates the shared
  tool schema that both platforms consume.
- **2 next**: prompt namespace extracts SystemPromptProvider — the prerequisite
  for the iOS port.
- **3 next**: message plumbing consolidates tool dispatch, reducing the code
  surface that Phase 2 just reorganized.
- **6**: interface cleanup — small scope, low risk.
- **4 last**: session filesystem — valuable but independent of the iOS port.

---

## What This Does NOT Include

- **9P protocol** — the JSON-lines stdio protocol already serves the same
  purpose with less ceremony. No benefit from switching.
- **rio / window management** — bubbletea handles terminal UI well. Plan 9's
  rio solves a different problem (multiplexed graphical terminals).
- **Factotum (auth delegation)** — HF_TOKEN and KAGI_API_KEY work fine as
  env vars. A separate auth service adds complexity without benefit at this scale.
- **CPU export (remote computation)** — the gateway mode (`httpProc`) already
  handles remote inference. No need for a general CPU export mechanism.
- **Union mounts** — tempting for layering project-local tools over global tools,
  but Go's embed + file loading already handles this for skills.json. The
  complexity of a virtual filesystem abstraction outweighs the benefit.

---

## Cross-Cutting: Shared Data Files

After all phases, the project gains three canonical data files:

```
models.json        — model catalog (Go + Python + Swift read this)
scripts/tools.json — tool schemas (Go + Python + Swift read this)
prompts/           — prompt templates by persona and tier (optional, Phase 2+)
```

These files embody the Plan 9 philosophy: configuration as readable text files
that any program can consume. They replace scattered constants, switch
statements, and duplicated definitions across languages.
