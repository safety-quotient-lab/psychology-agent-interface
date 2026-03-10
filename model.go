package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/style"
)

// appState is the TUI state machine.
type appState int

const (
	stateModelSelect appState = iota // interactive model picker
	stateLoading                     // waiting for sidecar ready handshake
	stateInput                       // textinput focused
	stateThinking                    // streaming inference in-flight
	stateToolRun                     // tool executing
	stateConfirm                     // awaiting y/n tool approval
	stateAskUser                     // waiting for user answer to ask_user tool
	stateReview                      // critic agent reviewing primary response
)

// context compaction constants
const (
	compactThreshold = 24 // compact when conversation exceeds this many messages
	compactKeep      = 16 // keep this many recent messages after compaction
)

// Custom tea.Msg types --------------------------------------------------------

type msgServerReady struct {
	model     string
	vramMB    int
	loadS     float64
	useNative bool
}

type msgServerError struct{ err error }

// msgToken carries one streamed token from the sidecar.
type msgToken struct{ token string }

// msgStreamDone signals end of a streaming inference.
// reply is non-empty for HTTP (non-streaming) backends.
type msgStreamDone struct {
	tokens  int
	elapsed float64
	err     error
	reply   string
}

type msgToolDone struct {
	name   string
	args   map[string]any
	result string
}

type msgShellDirect struct {
	cmd    string
	output string
}

type msgAutoContext struct{ fileList string }

// msgSidecarStarted carries the newly launched sidecar proc after model selection.
type msgSidecarStarted struct {
	proc inferProc
	err  error
}

type msgReviewToken struct{ token string }
type msgReviewDone struct {
	tokens  int
	elapsed float64
	err     error
	reply   string
}

// Message is a conversation turn sent to the sidecar.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

// modelInfo describes a selectable model for the picker UI.
type modelInfo struct {
	key   string // model key (e.g. "qwen-3b")
	label string // display label
	tier  string // psychology prompt tier
	vram  string // approximate VRAM usage
}

var selectableModels = []modelInfo{
	{"qwen-0.5b", "Qwen 2.5 0.5B", "Tier 1 (≤2B)", "~1 GB  ★ fast"},
	{"qwen-1.5b", "Qwen 2.5 1.5B", "Tier 1 (≤2B)", "~3 GB"},
	{"smollm2", "SmolLM2 1.7B", "Tier 1 (≤2B)", "~3 GB"},
	{"llama-1b", "Llama 3.2 1B", "Tier 1 (≤2B)", "~2 GB"},
	{"qwen-3b", "Qwen 2.5 3B", "Tier 2 (2B–4B)", "~5.6 GB"},
	{"gemma-2b", "Gemma 2 2B", "Tier 2 (2B–4B)", "~4 GB"},
	{"llama-3b", "Llama 3.2 3B", "Tier 2 (2B–4B)", "~6 GB"},
}

// Model is the bubbletea model. -----------------------------------------------

type Model struct {
	state     appState
	cfg       appConfig
	proc      inferProc
	useNative bool
	modelName string
	vramMB    int
	loadS     float64

	conversation []Message
	displayLines []string
	totalTokens  int
	totalTime    float64
	turnCount    int
	sessionTokens int

	// streaming accumulator — holds tokens until msgStreamDone
	// Using []byte avoids "illegal use of non-zero Builder copied by value"
	// since bubbletea copies the Model on every Update() call.
	streamBuf []byte

	// pending tool calls queued after an inference
	pendingTools []ToolCall
	toolIdx      int

	// pendingSummary: true while waiting for the compaction summary inference
	pendingSummary bool

	// Skills — named capability modes with system prompt overlays
	skills      []Skill
	activeSkill *Skill
	claudeMD    string // cached for system prompt rebuilds when skills change

	// Adversarial review — streaming accumulator for critic pass
	reviewBuf []byte

	// Tool output detail view — ctrl+o toggles full last result
	lastToolResult string
	showDetail     bool

	// model selector — interactive picker for startup and /model command
	selectorCursor int  // highlighted index in selectableModels
	selectorReturn bool // true when opened from /model (Esc returns to stateInput)

	// input history — shell-style up/down recall
	inputHistory []string
	historyIdx   int    // -1 = not browsing history
	inputDraft   string // saved current draft when history browsing begins

	spinner spinner.Model
	input   textinput.Model
	vp      viewport.Model
	width   int
	height  int
	vpReady bool
}

func newModel(c appConfig, proc inferProc) Model {
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(style.PurpleDim)

	ti := textinput.New()
	ti.Placeholder = "Type your message..."
	ti.CharLimit = 0

	return Model{state: stateLoading, cfg: c, proc: proc, spinner: sp, input: ti, historyIdx: -1,
		skills: loadSkills(c.cwd)}
}

// newModelSelector creates a Model starting in the model picker state (no proc yet).
func newModelSelector(c appConfig) Model {
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(style.PurpleDim)

	ti := textinput.New()
	ti.Placeholder = "Type your message..."
	ti.CharLimit = 0

	// Pre-select the default model in the cursor
	cursor := 0
	for i, m := range selectableModels {
		if m.key == c.model {
			cursor = i
			break
		}
	}

	return Model{state: stateModelSelect, cfg: c, spinner: sp, input: ti, historyIdx: -1,
		skills: loadSkills(c.cwd), selectorCursor: cursor}
}

// cmdStartSidecar launches the Python sidecar for the selected model.
func cmdStartSidecar(model, projectRoot, quant string) tea.Cmd {
	return func() tea.Msg {
		sp, err := startSidecarQuant(model, projectRoot, quant)
		if err != nil {
			return msgSidecarStarted{err: err}
		}
		return msgSidecarStarted{proc: sp}
	}
}

// Init fires the spinner and the blocking sidecar-ready read. -----------------

func (m Model) Init() tea.Cmd {
	if m.state == stateModelSelect {
		return m.spinner.Tick
	}
	return tea.Batch(m.spinner.Tick, cmdWaitReady(m.proc))
}

// tea.Cmd constructors --------------------------------------------------------

func cmdWaitReady(proc inferProc) tea.Cmd {
	return func() tea.Msg {
		msg, err := proc.waitReady()
		if err != nil {
			return msgServerError{err}
		}
		return msgServerReady{
			model: msg.Model, vramMB: msg.VramMB,
			loadS: msg.LoadS, useNative: msg.UseNative,
		}
	}
}


func readStreamLine(proc *llmProc) tea.Msg {
	if !proc.stdout.Scan() {
		return msgStreamDone{err: fmt.Errorf("sidecar closed")}
	}
	var line struct {
		Token   string  `json:"token"`
		Done    bool    `json:"done"`
		Tokens  int     `json:"tokens"`
		Seconds float64 `json:"seconds"`
		Error   string  `json:"error"`
	}
	if err := json.Unmarshal(proc.stdout.Bytes(), &line); err != nil {
		return msgStreamDone{err: fmt.Errorf("parse: %w", err)}
	}
	if line.Error != "" {
		return msgStreamDone{err: fmt.Errorf("%s", line.Error)}
	}
	if line.Done {
		return msgStreamDone{tokens: line.Tokens, elapsed: line.Seconds}
	}
	return msgToken{token: line.Token}
}

func cmdAutoContext(cwd string) tea.Cmd {
	return func() tea.Msg {
		result := executeTool("list_files", map[string]any{"pattern": "*"}, cwd)
		return msgAutoContext{fileList: result}
	}
}


func readReviewLine(proc *llmProc) tea.Msg {
	if !proc.stdout.Scan() {
		return msgReviewDone{err: fmt.Errorf("sidecar closed")}
	}
	var line struct {
		Token   string  `json:"token"`
		Done    bool    `json:"done"`
		Tokens  int     `json:"tokens"`
		Seconds float64 `json:"seconds"`
		Error   string  `json:"error"`
	}
	if err := json.Unmarshal(proc.stdout.Bytes(), &line); err != nil {
		return msgReviewDone{err: fmt.Errorf("parse: %w", err)}
	}
	if line.Error != "" {
		return msgReviewDone{err: fmt.Errorf("%s", line.Error)}
	}
	if line.Done {
		return msgReviewDone{tokens: line.Tokens, elapsed: line.Seconds}
	}
	return msgReviewToken{token: line.Token}
}

// criticPrompt builds the Socratic follow-up prompt.
// Instead of adversarial critique, generates deepening questions that guide
// the user toward their own insights — the core Socratic method loop.
func criticPrompt(userMsg, primaryReply string) string {
	return fmt.Sprintf(`You practice the Socratic method. Review the exchange below.
Generate 1-2 follow-up questions that:
- Surface unstated assumptions the user might hold
- Invite the user to examine their reasoning from a different angle
- Gently probe for evidence behind beliefs or conclusions

Do NOT lecture, advise, or evaluate. Only ask questions.
If the exchange reached a natural conclusion, reply with only "✓".

User said: %s

Assistant responded: %s

Your follow-up questions:`, userMsg, primaryReply)
}

func cmdShellDirect(text, cwd string) tea.Cmd {
	return func() tea.Msg {
		return msgShellDirect{cmd: text, output: runDirectShell(text, cwd)}
	}
}

func cmdTool(name string, args map[string]any, cwd string) tea.Cmd {
	return func() tea.Msg {
		return msgToolDone{name: name, args: args, result: executeTool(name, args, cwd)}
	}
}

// needsApproval returns true for tools that can modify state or run arbitrary code.
func needsApproval(name string) bool {
	return name == "shell" || name == "write_file" || name == "edit_file"
}

// loadClaudeFile walks from cwd toward home looking for CLAUDE.md; returns its content or "".
func loadClaudeFile(cwd string) string {
	home, _ := os.UserHomeDir()
	dir := cwd
	for {
		if data, err := os.ReadFile(filepath.Join(dir, "CLAUDE.md")); err == nil {
			return strings.TrimSpace(string(data))
		}
		parent := filepath.Dir(dir)
		if dir == home || parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

// buildToolPreview returns lines shown in the chat before the y/N confirm prompt.
func buildToolPreview(name string, args map[string]any, cwd string) string {
	switch name {
	case "write_file":
		path, _ := args["path"].(string)
		newContent, _ := args["content"].(string)
		if !filepath.IsAbs(path) {
			path = filepath.Join(cwd, path)
		}
		var oldSize int
		if data, err := os.ReadFile(path); err == nil {
			oldSize = len(data)
		}
		preview := firstNLines(newContent, 5)
		return style.Dim.Render(fmt.Sprintf("  %s  (%d → %d bytes)\n%s", path, oldSize, len(newContent), preview))
	case "edit_file":
		path, _ := args["path"].(string)
		oldStr, _ := args["old_str"].(string)
		newStr, _ := args["new_str"].(string)
		return style.Dim.Render(fmt.Sprintf("  %s\n--- replace ---\n%s\n--- with ---\n%s",
			path, firstNLines(oldStr, 4), firstNLines(newStr, 4)))
	}
	return ""
}

// truncateOutput returns the first n lines of s and whether truncation occurred.
func truncateOutput(s string, n int) (string, bool) {
	lines := strings.Split(s, "\n")
	if len(lines) <= n {
		return s, false
	}
	return strings.Join(lines[:n], "\n"), true
}

// firstNLines returns the first n lines of s, with "…" appended if truncated.
func firstNLines(s string, n int) string {
	lines := strings.SplitN(s, "\n", n+1)
	if len(lines) > n {
		lines = lines[:n]
		lines = append(lines, "…")
	}
	return strings.Join(lines, "\n")
}

// Update ----------------------------------------------------------------------

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.MouseMsg:
		if m.vpReady {
			var cmd tea.Cmd
			m.vp, cmd = m.vp.Update(msg)
			return m, cmd
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		if !m.vpReady {
			m.vp = viewport.New(msg.Width, m.chatHeight())
			m.vpReady = true
		} else {
			m.vp.Width = msg.Width
			m.vp.Height = m.chatHeight()
			m.syncViewport() // re-sync content to new width (fixes glitch during streaming)
		}
		m.syncViewport()
		return m, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd

	case msgServerReady:
		m.modelName = msg.model
		m.vramMB = msg.vramMB
		m.loadS = msg.loadS
		m.useNative = msg.useNative
		// Stay in stateLoading until msgAutoContext injects file context.
		// This prevents the race where the user submits before priming is done.
		claudeMD := loadClaudeFile(m.cfg.cwd)
		m.claudeMD = claudeMD
		if msg.useNative {
			m.conversation = []Message{{Role: "system", Content: nativeSystem(m.cfg.cwd, msg.model, "", claudeMD)}}
		} else {
			m.conversation = []Message{{Role: "system", Content: reactSystem(m.cfg.cwd, msg.model, "", claudeMD)}}
		}
		statusLine := fmt.Sprintf("Model loaded in %.1fs · %d MB VRAM · native=%v", msg.loadS, msg.vramMB, msg.useNative)
		if claudeMD != "" {
			statusLine += " · CLAUDE.md loaded"
		}
		m.displayLines = append(m.displayLines,
			style.Dim.Render(statusLine),
			style.Dim.Render("Indexing working directory..."),
		)
		m.syncViewport()
		return m, cmdAutoContext(m.cfg.cwd)

	case msgSidecarStarted:
		if msg.err != nil {
			m.displayLines = append(m.displayLines, style.Error.Render("failed to start sidecar: "+msg.err.Error()))
			m.state = stateModelSelect
			m.syncViewport()
			return m, nil
		}
		m.proc = msg.proc
		m.state = stateLoading
		m.displayLines = append(m.displayLines,
			style.Dim.Render(fmt.Sprintf("starting %s...", m.cfg.model)), "")
		m.syncViewport()
		return m, cmdWaitReady(m.proc)

	case msgServerError:
		m.displayLines = append(m.displayLines, style.Error.Render("ERROR: "+msg.err.Error()))
		m.state = stateInput
		m.syncViewport()
		return m, nil

	// ── Auto file context injected at startup ───────────────────────────────
	case msgAutoContext:
		fl := msg.fileList
		if fl == "(no matches)" || fl == "(no output)" {
			fl = ""
		}
		// Strip any existing primed list_files exchange before injecting the new one.
		// Without this, repeated cd/cwd/clear accumulate duplicate stale listings.
		m.conversation = stripPrimedContext(m.conversation)
		// Inject updated file listing as a primed tool-use exchange.
		if fl != "" {
			if m.useNative {
				m.conversation = append(m.conversation,
					Message{Role: "assistant", Content: `<tool_call>{"name":"list_files","arguments":{"pattern":"*"}}</tool_call>`},
					Message{Role: "tool", Name: "list_files", Content: fl},
				)
			} else {
				m.conversation = append(m.conversation,
					Message{Role: "user", Content: "List the files in the working directory."},
					Message{Role: "assistant", Content: `TOOL_CALL: {"name": "list_files", "arguments": {"pattern": "*"}}`},
					Message{Role: "user", Content: "TOOL_RESULT (list_files):\n" + fl + "\n\nContinue."},
					Message{Role: "assistant", Content: "Working directory contains: " + fl},
				)
			}
		}
		// Now allow user input
		m.state = stateInput
		m.input.Focus()
		m.displayLines = append(m.displayLines,
			style.Dim.Render("Type your task. /help for commands. Ctrl+C to exit."),
			"",
		)
		m.syncViewport()
		return m, nil

	// ── Streaming token ───────────────────────────────────────────────────────
	case msgToken:
		m.streamBuf = append(m.streamBuf, msg.token...)
		m.syncViewport()
		return m, m.proc.nextToken()

	// ── Streaming complete ────────────────────────────────────────────────────
	case msgStreamDone:
		// Drop stale messages from aborted inferences
		if m.state != stateThinking {
			return m, nil
		}
		// HTTP (non-streaming) backends deliver the full reply here.
		if msg.reply != "" {
			m.streamBuf = []byte(msg.reply)
		}

		if msg.err != nil {
			m.streamBuf = nil
			m.pendingSummary = false // clear in case error occurred mid-summary
			m.displayLines = append(m.displayLines, style.Error.Render("Error: "+msg.err.Error()), "")
			m.state = stateInput
			m.syncViewport()
			return m, nil
		}

		// Handle compaction summary inference
		if m.pendingSummary {
			m.pendingSummary = false
			summary := stripMarkup(string(m.streamBuf))
			m.streamBuf = nil
			if summary != "" {
				// Inject summary note just after system message (if any)
				note := Message{Role: "user", Content: "[Earlier context summary: " + summary + "]"}
				insertAt := 0
				if len(m.conversation) > 0 && m.conversation[0].Role == "system" {
					insertAt = 1
				}
				conv := make([]Message, 0, len(m.conversation)+1)
				conv = append(conv, m.conversation[:insertAt]...)
				conv = append(conv, note)
				conv = append(conv, m.conversation[insertAt:]...)
				m.conversation = conv
				restored := "cwd"
			if m.activeSkill != nil {
				restored += ", skill: " + m.activeSkill.ID
			}
			m.displayLines = append(m.displayLines,
				style.Dim.Render("[smart compact: "+summary+"]"),
				style.Dim.Render("[context restored: "+restored+"]"),
				"")
			}
			m.syncViewport()
			return m, m.proc.startInfer(m.conversation, 1024, 0.2)
		}

		m.sessionTokens += msg.tokens

		// Token budget check
		if m.cfg.tokenBudget > 0 && m.sessionTokens >= m.cfg.tokenBudget {
			m.streamBuf = nil
			m.displayLines = append(m.displayLines,
				style.Warning.Render(fmt.Sprintf("⚠ token budget reached (%d tokens)", m.sessionTokens)), "")
			m.state = stateInput
			m.syncViewport()
			return m, nil
		}

		reply := string(m.streamBuf)
		m.streamBuf = nil
		m.totalTokens += msg.tokens
		m.totalTime += msg.elapsed

		calls := m.parseCalls(reply)
		if len(calls) == 0 {
			clean := wordWrap(stripMarkup(reply), m.vp.Width-2)
			if clean == "" {
				clean = wordWrap(reply, m.vp.Width-2)
			}
			m.displayLines = append(m.displayLines,
				style.Value.Render(clean),
				style.Dim.Render(fmt.Sprintf("[%d tok · %.1fs]", m.totalTokens, m.totalTime)),
				"",
			)
			m.conversation = append(m.conversation, Message{Role: "assistant", Content: reply})
			if !m.cfg.noReview {
				userMsg := lastUserMsg(m.conversation)
				m.displayLines = append(m.displayLines, style.Dim.Render("─── critic ───"))
				m.state = stateReview
				m.syncViewport()
				return m, m.proc.startReview(userMsg, clean)
			}
			m.state = stateInput
			m.syncViewport()
			return m, nil
		}

		m.conversation = append(m.conversation, Message{Role: "assistant", Content: reply})
		m.pendingTools = calls
		m.toolIdx = 0
		cmd := m.dispatchNextTool()
		m.syncViewport()
		return m, cmd

	// ── Tool execution complete ───────────────────────────────────────────────
	case msgToolDone:
		w := m.width - 4
		if w < 20 {
			w = 20
		}
		argsJSON, _ := json.Marshal(msg.args)
		m.lastToolResult = msg.result
		m.showDetail = false

		const toolTruncLines = 10
		display, truncated := truncateOutput(msg.result, toolTruncLines)
		hint := ""
		if truncated {
			lines := strings.Count(msg.result, "\n") + 1
			hint = style.Dim.Render(fmt.Sprintf("  … +%d lines (ctrl+o to expand)", lines-toolTruncLines))
		}
		resultBox := lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(style.Green).
			Padding(0, 1).
			Width(w).
			Render(wordWrap(display, w-4))

		lines := []string{
			style.Warning.Render("["+msg.name+"]") + " " + style.Dim.Render(string(argsJSON)),
			resultBox,
		}
		if hint != "" {
			lines = append(lines, hint)
		}
		lines = append(lines, "")
		m.displayLines = append(m.displayLines, lines...)
		// Cap what goes into context — full result is kept in m.lastToolResult for display.
		const convResultLimit = 800
		convResult := msg.result
		if len(convResult) > convResultLimit {
			convResult = convResult[:convResultLimit] + "\n[truncated — use read_file to see full content]"
		}
		if m.useNative {
			m.conversation = append(m.conversation, Message{Role: "tool", Name: msg.name, Content: convResult})
		} else {
			m.conversation = append(m.conversation, Message{
				Role:    "user",
				Content: fmt.Sprintf("TOOL_RESULT (%s):\n%s\n\nContinue.", msg.name, convResult),
			})
		}

		m.toolIdx++
		if m.toolIdx < len(m.pendingTools) {
			cmd := m.dispatchNextTool()
			m.syncViewport()
			return m, cmd
		}
		return m, m.finishToolBatch()

	// ── Critic review token ───────────────────────────────────────────────────
	case msgReviewToken:
		m.reviewBuf = append(m.reviewBuf, msg.token...)
		m.syncViewport()
		return m, m.proc.nextReviewToken()

	// ── Critic review complete ────────────────────────────────────────────────
	case msgReviewDone:
		// HTTP backends deliver full review reply here.
		if msg.reply != "" {
			m.reviewBuf = []byte(msg.reply)
		}
		critique := strings.TrimSpace(string(m.reviewBuf))
		m.reviewBuf = nil
		if msg.err != nil {
			m.displayLines = append(m.displayLines,
				style.Dim.Render("[critic error: "+msg.err.Error()+"]"), "")
		} else if critique == "" || critique == "✓" {
			m.displayLines = append(m.displayLines, style.Success.Render("✓"), "")
		} else {
			m.displayLines = append(m.displayLines,
				style.Dim.Render(wordWrap(critique, m.vp.Width-2)), "")
		}
		m.state = stateInput
		m.syncViewport()
		return m, nil

	// ── Direct shell passthrough ──────────────────────────────────────────────
	case msgShellDirect:
		m.displayLines = append(m.displayLines,
			wordWrap(strings.TrimRight(msg.output, "\n"), m.vp.Width-2), "")
		m.state = stateInput
		m.syncViewport()
		return m, nil

	// ── Key events ───────────────────────────────────────────────────────────
	case tea.KeyMsg:
		if msg.Type == tea.KeyCtrlC {
			return m, tea.Quit
		}

		switch m.state {
		case stateModelSelect:
			switch msg.Type {
			case tea.KeyUp, tea.KeyShiftTab:
				if m.selectorCursor > 0 {
					m.selectorCursor--
				}
			case tea.KeyDown, tea.KeyTab:
				if m.selectorCursor < len(selectableModels)-1 {
					m.selectorCursor++
				}
			case tea.KeyEnter:
				selected := selectableModels[m.selectorCursor]
				m.cfg.model = selected.key
				if m.selectorReturn {
					// In-session switch: restart sidecar with new model
					m.selectorReturn = false
					if selected.key == m.modelName {
						m.displayLines = append(m.displayLines, style.Dim.Render("already using "+selected.key))
						m.state = stateInput
						m.syncViewport()
						return m, nil
					}
					proc, err := m.proc.restart(selected.key, m.cfg.projectRoot)
					if err != nil {
						m.displayLines = append(m.displayLines, style.Error.Render("failed to switch: "+err.Error()))
						m.state = stateInput
						m.syncViewport()
						return m, nil
					}
					m.proc = proc
					m.modelName = ""
					m.conversation = nil
					m.totalTokens = 0
					m.totalTime = 0
					m.turnCount = 0
					m.state = stateLoading
					m.displayLines = append(m.displayLines,
						style.Dim.Render(fmt.Sprintf("switching to %s...", selected.key)), "")
					m.syncViewport()
					return m, cmdWaitReady(m.proc)
				}
				// Startup flow: launch sidecar for the first time
				m.displayLines = append(m.displayLines,
					style.Dim.Render(fmt.Sprintf("starting %s...", selected.key)), "")
				m.syncViewport()
				return m, cmdStartSidecar(selected.key, m.cfg.projectRoot, m.cfg.quant)
			case tea.KeyEsc:
				if m.selectorReturn {
					// Cancel in-session picker, return to input
					m.selectorReturn = false
					m.state = stateInput
					m.input.Focus()
					return m, nil
				}
				// Startup: Esc quits
				return m, tea.Quit
			}
			return m, nil

		case stateThinking:
			if msg.Type == tea.KeyEsc {
				// Abort in-flight inference
				proc, err := m.proc.restart(m.cfg.model, m.cfg.projectRoot)
				if err == nil {
					m.proc = proc
				}
				// Remove last user message (was unresponded)
				if len(m.conversation) > 0 && m.conversation[len(m.conversation)-1].Role == "user" {
					m.conversation = m.conversation[:len(m.conversation)-1]
				}
				m.streamBuf = nil
				m.displayLines = append(m.displayLines, style.Warning.Render("⚠ aborted"), "")
				m.state = stateLoading
				m.syncViewport()
				return m, cmdWaitReady(m.proc)
			}

		case stateReview:
			if msg.Type == tea.KeyEsc {
				// Abort in-flight critic stream
				proc, err := m.proc.restart(m.cfg.model, m.cfg.projectRoot)
				if err == nil {
					m.proc = proc
				}
				m.reviewBuf = nil
				m.displayLines = append(m.displayLines, style.Dim.Render("[review skipped]"), "")
				m.state = stateLoading
				m.syncViewport()
				return m, cmdWaitReady(m.proc)
			}

		case stateInput:
			switch msg.Type {
			case tea.KeyEnter:
				text := strings.TrimSpace(m.input.Value())
				m.input.SetValue("")
				m.historyIdx = -1
				m.inputDraft = ""
				if text == "" {
					return m, nil
				}
				// Record in history (skip duplicates of last entry)
				if len(m.inputHistory) == 0 || m.inputHistory[len(m.inputHistory)-1] != text {
					m.inputHistory = append(m.inputHistory, text)
				}
				if strings.HasPrefix(text, "/") {
					cmd := m.handleSlash(text)
					m.syncViewport()
					return m, cmd
				}
				if cdCmd, done := m.handleCd(text); done {
					m.syncViewport()
					return m, cdCmd
				}
				if isShellCommand(text) {
					m.displayLines = append(m.displayLines, style.Dim.Render("$ "+text))
					m.state = stateToolRun
					m.syncViewport()
					return m, cmdShellDirect(text, m.cfg.cwd)
				}
				m.displayLines = append(m.displayLines, style.Heading.Render("> "+text), "")
				m.conversation = append(m.conversation, Message{Role: "user", Content: text})
				m.totalTokens = 0
				m.totalTime = 0
				m.turnCount = 0
				removed := m.compactIfNeeded()
				m.state = stateThinking
				m.syncViewport()
				if len(removed) > 0 {
					m.pendingSummary = true
					return m, m.proc.startInfer(makeSummaryMsgs(removed), 1024, 0.2)
				}
				return m, m.proc.startInfer(m.conversation, 1024, 0.2)

			case tea.KeyUp:
				// Scroll viewport if input is empty; otherwise browse history
				if m.input.Value() == "" || m.historyIdx != -1 || len(m.inputHistory) > 0 {
					if len(m.inputHistory) == 0 {
						break
					}
					if m.historyIdx == -1 {
						m.inputDraft = m.input.Value()
						m.historyIdx = len(m.inputHistory) - 1
					} else if m.historyIdx > 0 {
						m.historyIdx--
					}
					m.input.SetValue(m.inputHistory[m.historyIdx])
					m.input.CursorEnd()
					return m, nil
				}

			case tea.KeyDown:
				if m.historyIdx == -1 {
					break
				}
				m.historyIdx++
				if m.historyIdx >= len(m.inputHistory) {
					m.historyIdx = -1
					m.input.SetValue(m.inputDraft)
					m.inputDraft = ""
				} else {
					m.input.SetValue(m.inputHistory[m.historyIdx])
				}
				m.input.CursorEnd()
				return m, nil

			case tea.KeyCtrlO:
				// Toggle full tool output detail view
				if m.lastToolResult != "" {
					m.showDetail = !m.showDetail
					m.syncViewport()
				}
				return m, nil

			case tea.KeyPgUp, tea.KeyPgDown:
				// Scroll chat viewport without leaving input mode
				if m.vpReady {
					var cmd tea.Cmd
					m.vp, cmd = m.vp.Update(msg)
					return m, cmd
				}

			default:
				// Any typing while browsing history resets to draft
				if m.historyIdx != -1 {
					m.historyIdx = -1
					m.inputDraft = ""
				}
				var cmd tea.Cmd
				m.input, cmd = m.input.Update(msg)
				return m, cmd
			}

		case stateAskUser:
			if msg.Type == tea.KeyEnter {
				answer := strings.TrimSpace(m.input.Value())
				m.input.SetValue("")
				if answer == "" {
					return m, nil
				}
				m.displayLines = append(m.displayLines, style.Heading.Render("> "+answer), "")
				if m.useNative {
					m.conversation = append(m.conversation, Message{Role: "tool", Name: "ask_user", Content: answer})
				} else {
					m.conversation = append(m.conversation, Message{
						Role:    "user",
						Content: fmt.Sprintf("TOOL_RESULT (ask_user):\n%s\n\nContinue.", answer),
					})
				}
				m.toolIdx++
				if m.toolIdx < len(m.pendingTools) {
					cmd := m.dispatchNextTool()
					m.syncViewport()
					return m, cmd
				}
				return m, m.finishToolBatch()
			}
			var cmd tea.Cmd
			m.input, cmd = m.input.Update(msg)
			return m, cmd

		case stateConfirm:
			switch msg.String() {
			case "y", "Y", "enter":
				cmd := m.approveCurrentTool()
				m.syncViewport()
				return m, cmd
			case "n", "N", "esc":
				cmd := m.denyCurrentTool()
				m.syncViewport()
				return m, cmd
			}

		default:
			if m.vpReady {
				var cmd tea.Cmd
				m.vp, cmd = m.vp.Update(msg)
				return m, cmd
			}
		}
	}

	return m, nil
}

// Tool approval helpers -------------------------------------------------------

func (m *Model) dispatchNextTool() tea.Cmd {
	tool := m.pendingTools[m.toolIdx]
	// ask_user is handled interactively — no approval loop needed
	if tool.Name == "ask_user" {
		q, _ := tool.Args["question"].(string)
		m.displayLines = append(m.displayLines,
			style.Warning.Render("?  "+q),
			style.Dim.Render("  (type your answer and press Enter)"),
		)
		m.state = stateAskUser
		m.input.Focus()
		return nil
	}
	if !m.cfg.autoApprove && needsApproval(tool.Name) {
		if preview := buildToolPreview(tool.Name, tool.Args, m.cfg.cwd); preview != "" {
			m.displayLines = append(m.displayLines, preview)
		}
		m.state = stateConfirm
		return nil
	}
	m.state = stateToolRun
	return cmdTool(tool.Name, tool.Args, m.cfg.cwd)
}

func (m *Model) approveCurrentTool() tea.Cmd {
	tool := m.pendingTools[m.toolIdx]
	m.displayLines = append(m.displayLines, style.Success.Render("✓ approved"))
	m.state = stateToolRun
	return cmdTool(tool.Name, tool.Args, m.cfg.cwd)
}

func (m *Model) denyCurrentTool() tea.Cmd {
	name := m.pendingTools[m.toolIdx].Name
	m.displayLines = append(m.displayLines, style.Warning.Render("✗ denied: "+name), "")
	denial := fmt.Sprintf("Tool call '%s' was denied by the user.", name)
	if m.useNative {
		m.conversation = append(m.conversation, Message{Role: "tool", Name: name, Content: denial})
	} else {
		m.conversation = append(m.conversation, Message{
			Role:    "user",
			Content: fmt.Sprintf("TOOL_RESULT (%s):\n%s\n\nContinue.", name, denial),
		})
	}
	m.toolIdx++
	if m.toolIdx < len(m.pendingTools) {
		return m.dispatchNextTool()
	}
	return m.finishToolBatch()
}

func (m *Model) finishToolBatch() tea.Cmd {
	m.pendingTools = nil
	m.turnCount++
	if m.turnCount >= m.cfg.maxTurns {
		m.displayLines = append(m.displayLines,
			style.Warning.Render(fmt.Sprintf("Max turns (%d) reached.", m.cfg.maxTurns)), "")
		m.state = stateInput
		m.syncViewport()
		return nil
	}
	removed := m.compactIfNeeded()
	m.state = stateThinking
	m.syncViewport()
	if len(removed) > 0 {
		m.pendingSummary = true
		return m.proc.startInfer(makeSummaryMsgs(removed), 1024, 0.2)
	}
	return m.proc.startInfer(m.conversation, 1024, 0.2)
}

// Context compaction ----------------------------------------------------------

// compactIfNeeded trims the conversation if it exceeds compactThreshold.
// Returns the removed messages (nil if no compaction was needed).
func (m *Model) compactIfNeeded() []Message {
	if len(m.conversation) <= compactThreshold {
		return nil
	}
	start := 0
	if len(m.conversation) > 0 && m.conversation[0].Role == "system" {
		start = 1
	}
	if len(m.conversation)-start <= compactKeep {
		return nil
	}
	trimCount := len(m.conversation) - start - compactKeep
	removed := make([]Message, trimCount)
	copy(removed, m.conversation[start:start+trimCount])
	m.conversation = append(m.conversation[:start], m.conversation[start+trimCount:]...)
	m.displayLines = append(m.displayLines,
		style.Dim.Render(fmt.Sprintf("[context compacted — removed %d old messages, summarising...]", trimCount)), "")
	return removed
}

// makeSummaryMsgs builds a single-message conversation asking the LLM to summarise removed messages.
func makeSummaryMsgs(removed []Message) []Message {
	var sb strings.Builder
	sb.WriteString("Summarize this conversation history in 2-3 sentences:\n\n")
	for _, m := range removed {
		if m.Role == "user" || m.Role == "assistant" {
			clean := m.Content
			if m.Role == "assistant" {
				clean = stripMarkup(clean)
			}
			sb.WriteString(m.Role + ": " + clean + "\n")
		}
	}
	return []Message{{Role: "user", Content: sb.String()}}
}

// handleCd processes "cd <path>" input.
// Returns (cmd, true) if input was a cd command; (nil, false) otherwise.
// cmd is cmdAutoContext if cd succeeded (to refresh file listing), nil on error.
func (m *Model) handleCd(text string) (tea.Cmd, bool) {
	fields := strings.Fields(text)
	if len(fields) == 0 || fields[0] != "cd" {
		return nil, false
	}
	var target string
	if len(fields) == 1 {
		// bare "cd" → home directory
		home, err := os.UserHomeDir()
		if err != nil {
			m.displayLines = append(m.displayLines, style.Error.Render("cd: cannot determine home dir"))
			return nil, true
		}
		target = home
	} else {
		arg := strings.Join(fields[1:], " ")
		if filepath.IsAbs(arg) {
			target = arg
		} else {
			target = filepath.Join(m.cfg.cwd, arg)
		}
	}
	target = filepath.Clean(target)
	info, err := os.Stat(target)
	if err != nil || !info.IsDir() {
		m.displayLines = append(m.displayLines, style.Error.Render("cd: "+target+": no such directory"))
		return nil, true
	}
	m.cfg.cwd = target
	m.displayLines = append(m.displayLines, style.Dim.Render("cwd: "+m.cfg.cwd))
	return cmdAutoContext(m.cfg.cwd), true
}

// stripPrimedContext removes any previously injected list_files priming exchange
// from the conversation, identified by the sentinel assistant message.
func stripPrimedContext(conv []Message) []Message {
	const nativeSentinel = `<tool_call>{"name":"list_files","arguments":{"pattern":"*"}}</tool_call>`
	const reactSentinel  = "List the files in the working directory."
	out := conv[:0]
	skip := false
	for i, msg := range conv {
		if (msg.Role == "assistant" && msg.Content == nativeSentinel) ||
			(msg.Role == "user" && msg.Content == reactSentinel) {
			// Mark start of primed block; skip it and the messages that follow
			// (native: 2 messages; react: 4 messages)
			_ = i
			skip = true
		}
		if skip {
			// native block ends after tool result (Name=="list_files")
			// react block ends after assistant ack ("Working directory contains:")
			if (msg.Role == "tool" && msg.Name == "list_files") ||
				(msg.Role == "assistant" && strings.HasPrefix(msg.Content, "Working directory contains:")) {
				skip = false
			}
			continue
		}
		out = append(out, msg)
	}
	return out
}

// resetToolState clears any in-flight tool execution state.
// Call this when conversation is replaced (clear, session load) to avoid
// orphaned pendingTools executing against the new context.
func (m *Model) resetToolState() {
	m.pendingTools = nil
	m.toolIdx = 0
	m.pendingSummary = false
	m.streamBuf = nil
}

// Slash commands --------------------------------------------------------------

func (m *Model) handleSlash(text string) tea.Cmd {
	parts := strings.Fields(text)
	switch strings.ToLower(parts[0]) {

	case "/quit", "/exit":
		return tea.Quit

	case "/clear":
		m.resetToolState()
		m.conversation = nil
		claudeMD := loadClaudeFile(m.cfg.cwd)
		if m.useNative {
			m.conversation = []Message{{Role: "system", Content: nativeSystem(m.cfg.cwd, m.modelName, "", claudeMD)}}
		} else {
			m.conversation = []Message{{Role: "system", Content: reactSystem(m.cfg.cwd, m.modelName, "", claudeMD)}}
		}
		m.displayLines = []string{style.Dim.Render("(cleared)"), ""}
		m.totalTokens = 0
		m.totalTime = 0
		m.turnCount = 0
		return cmdAutoContext(m.cfg.cwd)

	case "/cwd":
		if len(parts) > 1 {
			m.cfg.cwd = parts[1]
			m.displayLines = append(m.displayLines, style.Dim.Render("cwd: "+m.cfg.cwd))
			return cmdAutoContext(m.cfg.cwd)
		}
		m.displayLines = append(m.displayLines, style.Dim.Render("cwd: "+m.cfg.cwd))

	case "/model":
		if len(parts) == 1 {
			// Open interactive model picker
			m.selectorReturn = true
			// Pre-select the current model
			for i, mi := range selectableModels {
				if mi.key == m.modelName {
					m.selectorCursor = i
					break
				}
			}
			m.state = stateModelSelect
			m.input.Blur()
			return nil
		}
		newModel := parts[1]
		valid := false
		for _, v := range validModels {
			if newModel == v {
				valid = true
				break
			}
		}
		if !valid {
			m.displayLines = append(m.displayLines, style.Error.Render(
				fmt.Sprintf("unknown model %q — valid: %s", newModel, strings.Join(validModels, ", "))))
			return nil
		}
		if newModel == m.modelName {
			m.displayLines = append(m.displayLines, style.Dim.Render("already using "+newModel))
			return nil
		}
		proc, err := m.proc.restart(newModel, m.cfg.projectRoot)
		if err != nil {
			m.displayLines = append(m.displayLines, style.Error.Render("failed to switch model: "+err.Error()))
			return nil
		}
		m.proc = proc
		m.cfg.model = newModel
		m.modelName = ""
		m.conversation = nil
		m.totalTokens = 0
		m.totalTime = 0
		m.turnCount = 0
		m.state = stateLoading
		m.displayLines = append(m.displayLines,
			style.Dim.Render(fmt.Sprintf("switching to %s...", newModel)), "")
		return cmdWaitReady(m.proc)

	case "/system":
		arg := strings.TrimSpace(strings.TrimPrefix(text, parts[0]))
		if arg == "" {
			// Show current system prompt
			if len(m.conversation) > 0 && m.conversation[0].Role == "system" {
				m.displayLines = append(m.displayLines,
					style.Dim.Render("system: "+m.conversation[0].Content))
			} else {
				m.displayLines = append(m.displayLines, style.Dim.Render("system: (none)"))
			}
		} else {
			// Set system prompt
			if len(m.conversation) > 0 && m.conversation[0].Role == "system" {
				m.conversation[0].Content = arg
			} else {
				m.conversation = append([]Message{{Role: "system", Content: arg}}, m.conversation...)
			}
			m.displayLines = append(m.displayLines,
				style.Dim.Render("system prompt updated"))
		}

	case "/export":
		path, err := exportMarkdown(m.modelName, m.conversation, m.cfg.cwd)
		if err != nil {
			m.displayLines = append(m.displayLines, style.Error.Render("export failed: "+err.Error()))
		} else {
			m.displayLines = append(m.displayLines, style.Dim.Render("exported: "+path))
		}

	case "/skill":
		m.handleSkill(parts)

	case "/session":
		m.handleSession(parts)

	case "/help":
		m.displayLines = append(m.displayLines,
			style.Dim.Render("/quit  /exit  /clear  /cwd <path>  /model [name or picker]"),
			style.Dim.Render("/system [text]  — show or set session system prompt"),
			style.Dim.Render("/export         — save conversation as markdown to cwd"),
			style.Dim.Render("/session save|list|load <n>  /help"),
			style.Dim.Render("/skill [id|list|off]  — invoke or list named skills"),
			style.Dim.Render("models: "+strings.Join(validModels, "  ")),
			style.Dim.Render("tools needing approval: shell, write_file  (bypass: --yes)"),
			style.Dim.Render("--budget N  — stop after N total tokens (0 = unlimited)"),
			style.Dim.Render("--no-review — disable adversarial critic pass after each response"),
			style.Dim.Render("ctrl+o      — expand/collapse last tool output"),
			style.Dim.Render("Escape      — abort in-flight inference/review and return to input"),
			style.Dim.Render("Up/Down     — cycle input history  ·  PgUp/PgDown — scroll chat"),
		)

	default:
		m.displayLines = append(m.displayLines, style.Warning.Render("Unknown command: "+parts[0]))
	}
	return nil
}

func (m *Model) handleSession(parts []string) {
	if len(parts) == 1 {
		m.displayLines = append(m.displayLines,
			style.Dim.Render("/session save  /session list  /session load <n>"))
		return
	}
	switch parts[1] {
	case "save":
		path, err := saveSession(m.modelName, m.conversation)
		if err != nil {
			m.displayLines = append(m.displayLines, style.Error.Render("save failed: "+err.Error()))
		} else {
			m.displayLines = append(m.displayLines, style.Dim.Render("saved: "+path))
		}
	case "list":
		sessions, err := listSessions()
		if err != nil {
			m.displayLines = append(m.displayLines, style.Error.Render("list failed: "+err.Error()))
		} else if len(sessions) == 0 {
			m.displayLines = append(m.displayLines, style.Dim.Render("no saved sessions"))
		} else {
			for i, s := range sessions {
				m.displayLines = append(m.displayLines,
					style.Dim.Render(fmt.Sprintf("  %d  %s  [%s]  %d msgs",
						i+1, s.ID, s.Model, len(s.Conversation))))
			}
		}
	case "load":
		if len(parts) < 3 {
			m.displayLines = append(m.displayLines, style.Warning.Render("usage: /session load <n>"))
			return
		}
		n, err := strconv.Atoi(parts[2])
		if err != nil {
			m.displayLines = append(m.displayLines, style.Warning.Render("invalid session number"))
			return
		}
		s, err := loadSessionByIndex(n)
		if err != nil {
			m.displayLines = append(m.displayLines, style.Error.Render("session not found"))
			return
		}
		m.resetToolState()
		m.conversation = s.Conversation
		m.displayLines = append(m.displayLines,
			style.Dim.Render(fmt.Sprintf("loaded: %s [%s] %d msgs", s.ID, s.Model, len(s.Conversation))))
		if s.Model != m.modelName {
			m.displayLines = append(m.displayLines,
				style.Warning.Render(fmt.Sprintf("⚠ session was saved with %s (running %s)", s.Model, m.modelName)))
		}
		m.displayLines = append(m.displayLines, "")
	default:
		m.displayLines = append(m.displayLines, style.Warning.Render("unknown: /session "+parts[1]))
	}
}

// Skills ----------------------------------------------------------------------

func (m *Model) handleSkill(parts []string) {
	if len(parts) == 1 || (len(parts) > 1 && parts[1] == "list") {
		if len(m.skills) == 0 {
			m.displayLines = append(m.displayLines,
				style.Dim.Render("no skills loaded"),
				style.Dim.Render("  create ~/.config/pai/skills.json or <cwd>/pai-skills.json"),
				style.Dim.Render(`  format: [{"id":"...","name":"...","description":"...","system_prompt":"..."}]`),
			)
			return
		}
		for _, s := range m.skills {
			active := ""
			if m.activeSkill != nil && m.activeSkill.ID == s.ID {
				active = " ✓"
			}
			m.displayLines = append(m.displayLines,
				style.Dim.Render(fmt.Sprintf("  %-22s %s%s", s.ID, s.Name, active)))
		}
		return
	}
	if len(parts) < 2 {
		return
	}
	if parts[1] == "off" {
		m.activeSkill = nil
		m.rebuildSystemPrompt()
		m.displayLines = append(m.displayLines, style.Dim.Render("skill deactivated"))
		return
	}
	skill := findSkill(m.skills, parts[1])
	if skill == nil {
		m.displayLines = append(m.displayLines, style.Warning.Render("skill not found: "+parts[1]))
		return
	}
	m.activeSkill = skill
	m.rebuildSystemPrompt()
	m.displayLines = append(m.displayLines,
		style.Dim.Render(fmt.Sprintf("skill active: %s — %s", skill.ID, skill.Name)))
}

// rebuildSystemPrompt regenerates conversation[0] from the base system prompt
// and the active skill overlay (if any). Call whenever skills change.
func (m *Model) rebuildSystemPrompt() {
	var base string
	if m.useNative {
		base = nativeSystem(m.cfg.cwd, m.modelName, "", m.claudeMD)
	} else {
		base = reactSystem(m.cfg.cwd, m.modelName, "", m.claudeMD)
	}
	if m.activeSkill != nil {
		base += "\n\n# Active skill: " + m.activeSkill.Name + "\n" + m.activeSkill.SystemPrompt
	}
	if len(m.conversation) > 0 && m.conversation[0].Role == "system" {
		m.conversation[0].Content = base
	} else {
		m.conversation = append([]Message{{Role: "system", Content: base}}, m.conversation...)
	}
}

// lastUserMsg returns the content of the most recent user message in conv.
func lastUserMsg(conv []Message) string {
	for i := len(conv) - 1; i >= 0; i-- {
		if conv[i].Role == "user" {
			return conv[i].Content
		}
	}
	return ""
}

// Viewport helpers ------------------------------------------------------------

func (m *Model) parseCalls(reply string) []ToolCall {
	if m.useNative {
		return parseNative(reply)
	}
	return parseReact(reply)
}

func (m *Model) syncViewport() {
	if !m.vpReady {
		return
	}
	m.vp.Height = m.chatHeight()

	// Track whether the user has scrolled up before we overwrite content.
	// During active streaming/review always follow the bottom; otherwise
	// respect the user's scroll position so they can read history.
	wasAtBottom := m.vp.AtBottom()
	activeStream := m.state == stateThinking || m.state == stateReview

	// ctrl+o detail view: show full last tool result instead of scrollback
	if m.showDetail && m.lastToolResult != "" {
		m.vp.SetContent(m.lastToolResult)
		return
	}

	content := strings.Join(m.displayLines, "\n")
	if len(m.streamBuf) > 0 {
		if content != "" {
			content += "\n"
		}
		content += style.Value.Render(string(m.streamBuf))
	}
	if len(m.reviewBuf) > 0 {
		if content != "" {
			content += "\n"
		}
		content += style.Dim.Render(string(m.reviewBuf))
	}
	m.vp.SetContent(content)
	if wasAtBottom || activeStream {
		m.vp.GotoBottom()
	}
}

func (m Model) chatHeight() int {
	inputLines := 1
	if m.state == stateConfirm {
		inputLines = 2 // tool detail + y/N prompt
	}
	if m.state == stateAskUser {
		inputLines = 1
	}
	h := m.height - 3 - inputLines
	if h < 1 {
		h = 1
	}
	return h
}

// View ------------------------------------------------------------------------

func (m Model) View() string {
	if !m.vpReady {
		if m.state == stateModelSelect {
			return m.renderModelSelector()
		}
		return m.spinner.View() + " " + style.Dim.Render("loading model...")
	}

	if m.state == stateModelSelect {
		return m.renderModelSelectorFull()
	}

	modelLabel := ""
	if m.modelName != "" {
		modelLabel = "[" + m.modelName + "]  "
	}
	header := style.Title.Width(m.width).Render(" pai  " + modelLabel + m.cfg.cwd + " ")

	var statusLine string
	switch m.state {
	case stateLoading:
		statusLine = m.spinner.View() + " " + style.Dim.Render("loading model...")
	case stateThinking:
		statusLine = m.spinner.View() + " " + style.Dim.Render("thinking...  (Esc to abort)")
	case stateReview:
		statusLine = m.spinner.View() + " " + style.Dim.Render("critic reviewing...  (Esc to skip)")
	case stateToolRun:
		statusLine = m.spinner.View() + " " + style.Dim.Render("running tool...")
	case stateConfirm:
		statusLine = m.spinner.View() + " " + style.Warning.Render("awaiting approval")
	case stateAskUser:
		statusLine = style.Warning.Render("? waiting for your answer")
	default:
		if m.showDetail {
			statusLine = style.Dim.Render("  [detail view — ctrl+o to close]")
		} else {
			statusLine = style.Dim.Render("  ")
		}
	}

	var inputBar string
	switch m.state {
	case stateInput:
		inputBar = style.Heading.Render("> ") + m.input.View()
	case stateAskUser:
		inputBar = style.Heading.Render("> ") + m.input.View()
	case stateConfirm:
		if len(m.pendingTools) > m.toolIdx {
			tool := m.pendingTools[m.toolIdx]
			argsJSON, _ := json.Marshal(tool.Args)
			inputBar = style.Warning.Render("["+tool.Name+"] ") + style.Dim.Render(string(argsJSON)) + "\n" +
				style.Heading.Render("Run this tool? [y/N] ")
		}
	}

	parts := []string{header, m.vp.View(), statusLine}
	if inputBar != "" {
		parts = append(parts, inputBar)
	}
	return strings.Join(parts, "\n")
}

// renderModelSelector renders the model picker before the viewport initializes.
func (m Model) renderModelSelector() string {
	return m.renderModelPicker(24) // use a reasonable default height
}

// renderModelSelectorFull renders the model picker using the full viewport dimensions.
func (m Model) renderModelSelectorFull() string {
	header := style.Title.Width(m.width).Render(" pai  select a model ")
	content := m.renderModelPicker(m.height - 4)
	hint := style.Dim.Render("  ↑/↓ navigate · Enter select · Esc " +
		func() string {
			if m.selectorReturn {
				return "cancel"
			}
			return "quit"
		}())
	return header + "\n" + content + "\n" + hint
}

// renderModelPicker renders the model list with the current cursor position.
func (m Model) renderModelPicker(maxHeight int) string {
	var b strings.Builder
	b.WriteString("\n")

	selectedStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#E0DEF4"))
	cursorStyle := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#C4A7E7"))
	dimStyle := style.Dim

	for i, mi := range selectableModels {
		cursor := "  "
		nameStyle := dimStyle
		if i == m.selectorCursor {
			cursor = cursorStyle.Render("> ")
			nameStyle = selectedStyle
		}

		line := fmt.Sprintf("%s%-22s %s   %s",
			cursor,
			nameStyle.Render(mi.label),
			dimStyle.Render(mi.tier),
			dimStyle.Render(mi.vram),
		)
		b.WriteString(line + "\n")

		if i >= maxHeight-4 {
			break
		}
	}
	b.WriteString("\n")
	return b.String()
}

// psychologyPromptTier selects the distilled psychology-agent system prompt
// based on model parameter count. Tiers from safety-quotient-lab/psychology-agent
// docs/prompts/ — behavioral directives sized for limited context windows.
func psychologyPromptTier(model string) string {
	// Tier 1: ≤2B params (qwen-0.5b, qwen-1.5b, llama-1b, smollm2)
	tier1 := `You are a psychology research assistant. You help with psychological analysis,
research methodology, and text interpretation. You do not diagnose, prescribe,
or deliver clinical judgments.

Rules:
1. Label every claim as [observation] or [inference]. Observations cite evidence.
   Inferences state the reasoning.
2. When uncertain, say "I am uncertain because..." before answering.
3. When a question falls outside psychology, say "This falls outside my scope."
4. Never agree just to be agreeable. If you disagree, state why.
5. Ask one clarifying question before long answers.

Format:
- Use short paragraphs (3-4 sentences max).
- End substantive answers with: "Confidence: high / moderate / low"
- If multiple interpretations exist, list them. Do not pick one silently.

Do not:
- Diagnose mental health conditions
- Claim clinical authority
- Fabricate citations or statistics
- Provide therapy or crisis intervention`

	// Tier 2: 2B-4B params (qwen-3b, llama-3b, gemma-2b, phi3-mini)
	tier2 := `You are the psychology agent — a collegial mentor for psychological analysis and
research. You advise; you do not decide. The user holds final authority.

Identity:
- Role: thinking partner, not authority. Guide toward discovery, never tell.
- Scope: psychology, research methodology, psychometric analysis, text safety.
- When near the edge of validated knowledge, say so explicitly.

Output discipline:
1. Separate observations from inferences. Use [OBS] and [INF] tags.
2. Link claims to evidence: "Based on [source], [claim]."
3. State uncertainty before conclusions: "Uncertainty: [what and why]."
4. When multiple interpretations exist, present the most parsimonious first.
5. Chunk responses into labeled sections. Never write walls of text.
6. End with: "Confidence: HIGH / MODERATE / LOW — [one-line basis]"

Hard refusals:
- Never diagnose. PSQ scores text, not people.
- Never fabricate confidence where evidence lacks.
- Never soften a position without stating what new evidence justified the change.
- Never average conflicting sources — report the disagreement.
- Never provide crisis intervention (direct to 988 Suicide & Crisis Lifeline).

When disagreeing with the user:
- State the evidence for your position.
- Ask: "What evidence would change my assessment?"
- If no new evidence appears, hold your position respectfully.`

	// Tier 3: 4B-8B params (qwen-7b, llama-8b, mistral-7b)
	tier3 := `You are the psychology agent — a collegial mentor who synthesizes across
psychology, research methodology, and engineering. Your role: advisory,
Socratic, discipline-first. The user decides; you analyze and challenge.

Core stance:
- Socratic: ask before concluding. Generate competing hypotheses before settling.
- Anti-sycophancy: hold positions under pushback unless new evidence justifies
  updating. If you update, name what changed.
- Fair witness: report what happened, not why. Separate facts from conclusions.
- Recommend-against: before any default action, scan for a concrete reason NOT
  to proceed. Surface it if found.

Output discipline:
1. [OBS] for observations (directly evidenced). [INF] for inferences (reasoning).
2. Link every claim to evidence. Unsupported claims get flagged with ⚑.
3. State uncertainty dimensions before conclusions.
4. Parsimony first: prefer the interpretation with fewer assumptions.
5. Chunk into labeled sections. Offer stopping points for long answers.
6. Confidence footer: "Confidence: HIGH/MOD/LOW — [basis]. Evidence quality:
   HIGH/MOD/LOW/VERY LOW."

Interpretant awareness:
- When a term has multiple meanings across communities (clinical vs statistical
  vs lay), bind which meaning you intend before using it.
- When the user's vocabulary shifts mid-conversation, note the shift.

Scope boundaries:
- Psychology, psychometrics, research methodology: respond fully.
- Adjacent domains (law, clinical practice, engineering): reason but flag as
  inference, not expertise.
- Outside scope: acknowledge and redirect.

Hard refusals:
- Never diagnose. Never deliver verdicts. Never fabricate confidence.
- Never compress disagreement into consensus. Report the shape of conflict.
- Never provide crisis intervention (direct to 988 Lifeline / local equivalent).
- Never adopt a persona that suspends epistemic discipline.`

	switch model {
	case "qwen-3b", "llama-3b", "gemma-2b":
		return tier2
	case "qwen-7b", "llama-8b", "mistral-7b":
		return tier3
	default:
		// ≤2B: qwen-0.5b, qwen-1.5b, llama-1b, smollm2, and unknown models
		return tier1
	}
}

// reactSystem returns the system prompt for models without native tool calling.
// Combines the psychology-agent identity prompt with tool-calling instructions.
func reactSystem(cwd, model, fileList, claudeMD string) string {
	s := psychologyPromptTier(model) + `

Working directory: ` + cwd + `

You also have access to local tools for investigating files and running commands.

Available tools:
- shell(cmd): Execute a bash command. Returns combined stdout and stderr.
- read_file(path): Read the full contents of a file.
- write_file(path, content): Write content to a file (overwrites if it exists).
- edit_file(path, old_str, new_str): Replace the first occurrence of old_str with new_str. old_str must match exactly (whitespace matters). Prefer this over write_file for targeted edits.
- list_files(pattern): List files matching a glob pattern. Use "*.ext" for current dir, "**/*.ext" to recurse.
- search(pattern): Search for a regex pattern in files (like grep -rn).
- fetch_url(url): Fetch a URL and return its content as plain text. Use for docs, GitHub issues, paste links.
- web_search(query, [limit]): Search the web via Kagi. Returns titles, URLs, snippets. Default limit 5.
- ask_user(question): Ask the user a clarifying question and get their answer.

To call a tool, output EXACTLY this format on its own line (nothing else on that line):
TOOL_CALL: {"name": "tool_name", "arguments": {"arg": "value"}}

After each TOOL_CALL line you receive a TOOL_RESULT line. Use as many tool calls as needed.
When finished, write your final answer with no TOOL_CALL lines.`
	if fileList != "" {
		s += "\n\nFiles in working directory:\n" + fileList
	}
	if claudeMD != "" {
		s += "\n\n# Project instructions\n" + claudeMD
	}
	return s
}

// nativeSystem returns the system prompt for models with native tool calling.
// Combines psychology-agent identity with tool orientation.
func nativeSystem(cwd, model, fileList, claudeMD string) string {
	s := psychologyPromptTier(model) + `

Working directory: ` + cwd + `
You have tools: shell, read_file, write_file, edit_file, list_files, search, fetch_url, web_search, ask_user.
Use tools when you need to investigate files or run commands.`
	if fileList != "" {
		s += "\n\nFiles in working directory:\n" + fileList
	}
	if claudeMD != "" {
		s += "\n\n# Project instructions\n" + claudeMD
	}
	return s
}
