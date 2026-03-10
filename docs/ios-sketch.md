# pai iOS — Design Sketch

Status: **draft** — architectural sketch for pickup later.

---

## 1. Inference Engine

**llama.cpp** via C API from Swift. Not MLX Swift (macOS-focused, less iOS testing).

Reference: `llama.cpp/examples/llama.swiftui/` — Apple ships a working SwiftUI example.

### Model format

Convert HuggingFace safetensors → GGUF (Q4_K_M quantization).

| Model | GGUF size | iPhone target |
|-------|-----------|---------------|
| qwen-0.5b | ~350 MB | All (bundle candidate) |
| qwen-1.5b | ~900 MB | All (download) |
| qwen-3b | ~2 GB | 8 GB devices only |
| llama-1b | ~700 MB | All (download) |
| llama-3b | ~2 GB | 8 GB devices only |

Download on first launch, cache in `Library/Caches`. Do not bundle in app binary.

### Memory budget

Gate model selection by `os_proc_available_memory()`. iPhone kills apps exceeding
~4–5 GB on 8 GB devices, ~3 GB on 6 GB devices. KV cache adds ~200–500 MB on top
of model weights.

---

## 2. SystemPromptProvider Protocol

The core abstraction — decouples prompt content from inference and UI.

```swift
// MARK: - Protocol

/// Configures identity, rules, and tool-calling format for a session.
protocol SystemPromptProvider {
    /// Display name for the UI (e.g. "Psychology Agent")
    var name: String { get }

    /// Base identity + behavioral rules, sized for model capacity.
    func identityPrompt(for tier: ModelTier) -> String

    /// Tool descriptions filtered by platform support.
    func toolDescriptions(capabilities: PlatformCapabilities) -> [ToolDescription]

    /// Assemble the final system prompt.
    func build(
        tier: ModelTier,
        capabilities: PlatformCapabilities,
        context: SessionContext
    ) -> String
}

// MARK: - Supporting types

enum ModelTier: Int, CaseIterable {
    case tier1 = 1  // ≤2B params
    case tier2 = 2  // 2B–4B params
    case tier3 = 3  // 4B–8B params

    init(model: String) {
        switch model {
        case "qwen-3b", "llama-3b", "gemma-2b": self = .tier2
        case "qwen-7b", "llama-8b", "mistral-7b": self = .tier3
        default: self = .tier1
        }
    }
}

struct PlatformCapabilities {
    var hasShell: Bool              // false on iOS
    var hasFileSystem: Bool         // sandboxed on iOS
    var hasWebSearch: Bool          // true when KAGI key in Keychain
    var hasNativeToolCalling: Bool  // true for Qwen models
    var workingDirectory: String?   // nil on iOS
}

struct SessionContext {
    var fileList: String?
    var projectInstructions: String?
}

struct ToolDescription {
    var name: String       // "read_file"
    var signature: String  // "read_file(path)"
    var summary: String    // "Read the full contents of a file."
}
```

### Go-side mirror (refactor existing code)

```go
type SystemPromptProvider interface {
    Name() string
    IdentityPrompt(tier int) string
    ToolDescriptions(caps PlatformCapabilities) []ToolDescription
    Build(tier int, caps PlatformCapabilities, ctx SessionContext) string
}
```

The existing `psychologyPromptTier()`, `reactSystem()`, `nativeSystem()` collapse
into a `PsychologyAgentProvider` struct implementing this interface. Other providers
(generic assistant, research methods, custom) plug in without touching inference or
tool-calling code.

---

## 3. Tool Mapping

| Tool | macOS | iOS | Notes |
|------|-------|-----|-------|
| `ask_user` | direct | direct | SwiftUI alert / inline prompt |
| `fetch_url` | `http.Client` | `URLSession` | identical behavior |
| `web_search` | Kagi API | Kagi API | key stored in Keychain on iOS |
| `read_file` | full FS | sandboxed | document picker imports |
| `write_file` | full FS | sandboxed | app Documents only |
| `edit_file` | full FS | sandboxed | app Documents only |
| `list_files` | doublestar glob | `FileManager.enumerator` | sandbox dirs only |
| `search` | `rg` subprocess | `NSRegularExpression` | scan sandbox files |
| `shell` | `exec.Command` | **removed** | no shell on iOS |

`PlatformCapabilities` drives which tools appear in the prompt. The tool executor
rejects calls to unavailable tools with a non-retryable error message.

---

## 4. App Structure

```
PaiApp/
├── App/
│   └── PaiApp.swift                 -- @main, scene setup
├── Views/
│   ├── ChatView.swift               -- conversation + streaming display
│   ├── ModelPickerView.swift         -- model selector overlay
│   ├── ToolApprovalSheet.swift       -- confirm/deny tool calls
│   ├── AskUserSheet.swift           -- inline user prompt
│   ├── MessageBubble.swift          -- glass bubble component
│   └── ToolCallCard.swift           -- compact expandable tool card
├── State/
│   ├── AppState.swift               -- @Observable, main state machine
│   ├── Conversation.swift           -- [Message], compaction, history
│   └── ModelCatalog.swift           -- available models, tier map, RAM check
├── Inference/
│   ├── InferenceEngine.swift        -- protocol (mirrors Go inferProc)
│   ├── LlamaCppEngine.swift         -- llama.cpp C-API wrapper
│   ├── RemoteEngine.swift           -- HTTP gateway mode
│   └── ModelDownloader.swift        -- download + cache GGUF files
├── Tools/
│   ├── ToolExecutor.swift           -- dispatch + sandbox enforcement
│   ├── ToolParser.swift             -- parseNative + parseReact
│   └── SandboxFileManager.swift     -- scoped file ops
├── Prompts/
│   ├── SystemPromptProvider.swift   -- protocol + supporting types
│   ├── PsychologyAgent.swift        -- tier 1/2/3 prompts
│   └── CriticPrompt.swift           -- Socratic follow-up
├── Persistence/
│   ├── SessionExporter.swift        -- JSONL export, share sheet
│   └── KeychainHelper.swift         -- API key storage
└── Resources/
    └── (no bundled models — download on demand)
```

---

## 5. State Machine

Maps from Go bubbletea states to Swift `@Observable`:

```swift
@Observable
final class AppState {
    enum Phase {
        case modelSelect        // picker overlay
        case loading            // model loading / downloading
        case input              // user typing
        case thinking           // inference streaming
        case toolRun            // executing a tool
        case toolConfirm        // awaiting user approval
        case askUser            // ask_user prompt displayed
        case review             // Socratic critic pass
    }

    var phase: Phase = .modelSelect
    var conversation: [Message] = []
    var streamingBuffer: String = ""
    var pendingTools: [ToolCall] = []
    var currentToolIndex: Int = 0
    var selectedModel: ModelInfo?

    // Token stream from inference engine
    func startInference(_ input: String) async { ... }
    // Tool dispatch
    func dispatchNextTool() async { ... }
}
```

---

## 6. Glass UI Spec

### Materials & layers

- **Chat background**: subtle gradient or `.ultraThinMaterial` over wallpaper-style base
- **User bubbles**: `.thinMaterial` + accent color tint, `ContainerRelativeShape`, 16pt corners
- **Agent bubbles**: `.ultraThinMaterial`, no tint, same shape
- **Tool cards**: `.regularMaterial`, SF Symbol leading icon, expandable via `DisclosureGroup`
- **Input bar**: `.thickMaterial` strip pinned to bottom, capsule-shaped `TextField`
- **Model picker**: full-screen `.ultraThinMaterial` overlay with spring presentation
- **Navigation bar**: inline title, blur-backed (default iOS behavior when using materials)

### Typography

- Body: `.body` Dynamic Type, primary label color
- Over materials: use `.secondary` vibrancy for readability
- Code blocks: `.monospaced` variant, slightly tinted glass background
- Confidence tags: `.caption` weight, muted color

### Motion

- Sheet presentations: `.spring(response: 0.35, dampingFraction: 0.85)`
- Streaming tokens: `opacity` transition, 0.15s ease-in
- Tool cards: `matchedGeometryEffect` expand/collapse
- Phase transitions: `withAnimation(.smooth)` on `phase` changes

### SF Symbols

- `brain.head.profile` — psychology agent identity
- `arrow.down.doc` — model download
- `terminal` — tool execution
- `checkmark.shield` — tool approval
- `magnifyingglass` — web search
- `doc.text` — file operations
- `bubble.left.and.bubble.right` — conversation
- `questionmark.circle` — ask_user

### Accessibility

- All materials adapt to Reduce Transparency setting (falls back to opaque)
- Dynamic Type throughout — no hardcoded font sizes
- VoiceOver labels on all tool cards and action buttons
- Minimum 44pt tap targets

---

## 7. InferenceEngine Protocol

```swift
/// Mirrors the Go inferProc interface.
protocol InferenceEngine {
    /// Block until the engine accepts requests.
    func waitReady() async throws

    /// Full (non-streaming) inference. Returns complete response.
    func infer(messages: [Message], system: String) async throws -> String

    /// Start streaming inference. Returns an AsyncStream of tokens.
    func startInfer(messages: [Message], system: String) async throws -> AsyncStream<String>

    /// Swap to a different model.
    func restart(model: String) async throws

    /// Release resources.
    func shutdown() async
}
```

Two implementations:
- `LlamaCppEngine` — on-device, wraps `llama.h` C API
- `RemoteEngine` — HTTP gateway mode (same as Go `httpProc`)

---

## 8. MVP Phases

### Phase 1 — Conversational core
- [ ] Xcode project, SwiftUI app scaffold
- [ ] `SystemPromptProvider` protocol + `PsychologyAgent` implementation
- [ ] `InferenceEngine` protocol + `LlamaCppEngine` (wrapping llama.cpp C API)
- [ ] `ModelDownloader` — fetch GGUF from hosted URL, progress UI
- [ ] `ChatView` with glass bubbles, streaming token display
- [ ] `ModelPickerView` with RAM-gated model list
- [ ] `AppState` state machine
- [ ] `ask_user` tool only
- [ ] Socratic critic loop

### Phase 2 — Extended tools
- [ ] `fetch_url` via URLSession
- [ ] `web_search` via Kagi API, key in Keychain
- [ ] Document picker → `read_file`
- [ ] `ToolApprovalSheet` for write operations
- [ ] Session export (JSONL + share sheet)

### Phase 3 — Full parity
- [ ] Sandboxed `write_file`, `edit_file`, `list_files`, `search`
- [ ] Skills system
- [ ] `RemoteEngine` (HTTP gateway mode)
- [ ] Context compaction
- [ ] Input history (swipe up for previous messages)

---

## 9. Go Refactor Checklist

Before or alongside the iOS work, refactor the Go codebase to share the
`SystemPromptProvider` abstraction:

- [ ] Extract `SystemPromptProvider` interface in `prompt.go`
- [ ] Move `psychologyPromptTier` into `PsychologyAgentProvider` struct
- [ ] Move `reactSystem` / `nativeSystem` into provider `.Build()` method
- [ ] `PlatformCapabilities` struct replaces ad-hoc `os.Getenv` checks
- [ ] Tool descriptions become data (slice of structs), not string concatenation
- [ ] Provider selection via config flag or startup picker

---

## 10. Open Questions

- Host GGUF models where? HuggingFace Hub works (direct download URLs). Or self-host.
- App Store review: Apple may scrutinize on-device LLM apps. Test with TestFlight first.
- Minimum iOS version: 17.0 (for `@Observable`, modern SwiftUI materials).
- Should the Go version also gain a `PlatformCapabilities` concept now, or wait?
- Model fine-tuning: can we ship LoRA adapters for psychology-specific behavior?
