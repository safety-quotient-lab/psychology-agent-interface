package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/reflow/wordwrap"
	plog "github.com/safety-quotient-lab/psychology-agent-interface/pkg/log"
	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/prompt"
	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/session"
	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/style"
)

// ── Keybindings ──────────────────────────────────────────────────────────────

type keyBindings struct {
	Submit   key.Binding
	Newline  key.Binding
	Abort    key.Binding
	Quit     key.Binding
	Detail   key.Binding
	History  key.Binding
	Scroll   key.Binding
	Approve  key.Binding
	Deny     key.Binding
	Select   key.Binding
	Navigate key.Binding
}

func newKeyBindings() keyBindings {
	return keyBindings{
		Submit:   key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "send")),
		Newline:  key.NewBinding(key.WithKeys("shift+enter"), key.WithHelp("shift+enter", "newline")),
		Abort:    key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "abort")),
		Quit:     key.NewBinding(key.WithKeys("ctrl+c"), key.WithHelp("ctrl+c", "quit")),
		Detail:   key.NewBinding(key.WithKeys("ctrl+o"), key.WithHelp("ctrl+o", "detail")),
		History:  key.NewBinding(key.WithKeys("up", "down"), key.WithHelp("↑/↓", "history")),
		Scroll:   key.NewBinding(key.WithKeys("pgup", "pgdown"), key.WithHelp("pgup/dn", "scroll")),
		Approve:  key.NewBinding(key.WithKeys("y"), key.WithHelp("y/n", "approve/deny")),
		Deny:     key.NewBinding(key.WithKeys("n"), key.WithHelp("n", "deny")),
		Select:   key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "select")),
		Navigate: key.NewBinding(key.WithKeys("up", "down"), key.WithHelp("↑/↓", "navigate")),
	}
}

// stateHelp returns the keybinding hints for the current state.
func (m Model) stateHelp() []key.Binding {
	switch m.state {
	case stateInput:
		return []key.Binding{m.keys.Submit, m.keys.Newline, m.keys.History, m.keys.Detail, m.keys.Scroll, m.keys.Quit}
	case stateThinking:
		return []key.Binding{m.keys.Abort, m.keys.Quit}
	case stateReview:
		return []key.Binding{m.keys.Abort, m.keys.Quit}
	case stateConfirm:
		return []key.Binding{m.keys.Approve, m.keys.Quit}
	case stateAskUser:
		return []key.Binding{m.keys.Submit, m.keys.Quit}
	case stateModelSelect:
		return []key.Binding{m.keys.Navigate, m.keys.Select, m.keys.Abort, m.keys.Quit}
	default:
		return []key.Binding{m.keys.Quit}
	}
}

// helpKeyMap adapts stateHelp for the help.Model interface.
type helpKeyMap struct{ bindings []key.Binding }

func (h helpKeyMap) ShortHelp() []key.Binding { return h.bindings }
func (h helpKeyMap) FullHelp() [][]key.Binding { return [][]key.Binding{h.bindings} }

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
	compactThreshold = 24   // compact when conversation exceeds this many messages
	compactKeep      = 16   // keep this many recent messages after compaction
	tokenReserve     = 1200 // tokens reserved for model response + overhead
)

// Context limits and tier info now live in models.json, accessed via cfg.catalog.

// modelTier returns the tier number for the current model from the catalog.
func (m Model) modelTier() int { return m.cfg.catalog.Tier(m.modelName) }

// Custom tea.Msg types --------------------------------------------------------

type msgServerReady struct {
	model     string
	vramMB    int
	loadS     float64
	useNative bool
}

type msgServerError struct{ err error }

// msgLoadProgress carries incremental loading milestones from the sidecar.
type msgLoadProgress struct {
	pct   float64
	stage string
}

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

// Model metadata (selectable models, context limits, tier info) now lives in
// models.json and loads into cfg.catalog at startup. See pkg/catalog.

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
	ns          prompt.Namespace // prompt namespace (Plan 9 Phase 2)

	// Adversarial review — streaming accumulator for critic pass
	reviewBuf []byte

	// Tool output detail view — ctrl+o toggles full last result
	lastToolResult string
	showDetail     bool

	// model selector — interactive picker for startup and /model command
	selectorCursor int    // highlighted index in catalog.All()
	selectorReturn bool   // true when opened from /model (Esc returns to stateInput)
	selectorError  string // error message displayed above model list

	// input history — shell-style up/down recall
	inputHistory []string
	historyIdx   int    // -1 = not browsing history
	inputDraft   string // saved current draft when history browsing begins

	spinner  spinner.Model
	input    textarea.Model
	vp       viewport.Model
	help     help.Model
	keys     keyBindings
	progress progress.Model
	loadStage string // current loading stage label
	width    int
	height   int
	vpReady  bool
}

func newTextarea() textarea.Model {
	ta := textarea.New()
	ta.Placeholder = "Type your message... (Shift+Enter for newline)"
	ta.CharLimit = 0
	ta.SetHeight(3)
	ta.ShowLineNumbers = false
	ta.FocusedStyle.CursorLine = lipgloss.NewStyle() // no highlight on current line
	ta.KeyMap.InsertNewline.SetKeys("shift+enter")
	return ta
}

func newProgressBar() progress.Model {
	return progress.New(
		progress.WithDefaultGradient(),
		progress.WithoutPercentage(),
	)
}

func newModel(c appConfig, proc inferProc) Model {
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(style.PurpleDim)

	h := help.New()
	h.Styles.ShortKey = lipgloss.NewStyle().Foreground(style.Gray)
	h.Styles.ShortDesc = lipgloss.NewStyle().Foreground(style.DimGray)

	return Model{state: stateLoading, cfg: c, proc: proc, spinner: sp,
		input: newTextarea(), help: h, keys: newKeyBindings(),
		progress: newProgressBar(), historyIdx: -1,
		skills: loadSkills(c.cwd)}
}

// newModelSelector creates a Model starting in the model picker state (no proc yet).
func newModelSelector(c appConfig) Model {
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = lipgloss.NewStyle().Foreground(style.PurpleDim)

	h := help.New()
	h.Styles.ShortKey = lipgloss.NewStyle().Foreground(style.Gray)
	h.Styles.ShortDesc = lipgloss.NewStyle().Foreground(style.DimGray)

	// Pre-select the default model in the cursor
	cursor := 0
	for i, m := range c.catalog.All() {
		if m.Key == c.model {
			cursor = i
			break
		}
	}

	return Model{state: stateModelSelect, cfg: c, spinner: sp,
		input: newTextarea(), help: h, keys: newKeyBindings(),
		progress: newProgressBar(), historyIdx: -1,
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
	return tea.Batch(m.spinner.Tick, cmdReadLoadLine(m.proc))
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

// cmdReadLoadLine reads one JSON line from the sidecar during loading.
// Progress lines become msgLoadProgress; the ready line becomes msgServerReady.
// For non-llmProc backends, falls back to blocking cmdWaitReady.
func cmdReadLoadLine(proc inferProc) tea.Cmd {
	lp, ok := proc.(*llmProc)
	if !ok {
		return cmdWaitReady(proc)
	}
	return func() tea.Msg {
		if !lp.stdout.Scan() {
			return msgServerError{fmt.Errorf("sidecar closed during load: %s", lp.lastStderrLine())}
		}
		raw := lp.stdout.Bytes()
		// Try progress line first
		var prog struct {
			LoadingPct float64 `json:"loading_pct"`
			Stage      string  `json:"stage"`
		}
		if json.Unmarshal(raw, &prog) == nil && prog.Stage != "" {
			return msgLoadProgress{pct: prog.LoadingPct, stage: prog.Stage}
		}
		// Otherwise expect the ready handshake
		var rp readyPayload
		if err := json.Unmarshal(raw, &rp); err != nil {
			return msgServerError{fmt.Errorf("parse load line: %w", err)}
		}
		return msgServerReady{
			model: rp.Model, vramMB: rp.VramMB,
			loadS: rp.LoadS, useNative: rp.UseNative,
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

// initNamespace sets up the prompt namespace with psychology-agent providers.
// Call after model loads and whenever context changes (cwd, skills, claudeMD).
func (m *Model) initNamespace() {
	m.ns = prompt.Namespace{
		Identity: prompt.PsychologyIdentity{},
		Tools:    prompt.DefaultToolProvider(),
		Context:  &prompt.WorkspaceContext{CWD: m.cfg.cwd, ClaudeMD: m.claudeMD},
		Format:   prompt.PsychologyFormat{},
		FewShot:  prompt.PsychologyFewShot{},
	}
	if m.activeSkill != nil {
		m.ns.Skill = &prompt.SkillOverlay{Name: m.activeSkill.Name, SystemPrompt: m.activeSkill.SystemPrompt}
	}
}

// buildSystemPrompt returns the system prompt for the current namespace and mode.
func (m *Model) buildSystemPrompt() string {
	sys, _ := m.ns.Build(m.modelTier(), m.useNative)
	return sys
}

// buildPriming returns few-shot priming messages for the current tier.
func (m *Model) buildPriming() []prompt.Message {
	_, priming := m.ns.Build(m.modelTier(), m.useNative)
	return priming
}

// sessionMsgsFromMain converts main-package Message slices to session.Message slices.
func sessionMsgsFromMain(msgs []Message) []session.Message {
	out := make([]session.Message, len(msgs))
	for i, m := range msgs {
		out[i] = session.Message{Role: m.Role, Content: m.Content, Name: m.Name}
	}
	return out
}

// sessionMsgsToMain converts session.Message slices to main-package Message slices.
func sessionMsgsToMain(smsgs []session.Message) []Message {
	out := make([]Message, len(smsgs))
	for i, sm := range smsgs {
		out[i] = Message{Role: sm.Role, Content: sm.Content, Name: sm.Name}
	}
	return out
}

// promptMsgsToMain converts prompt.Message slices to main-package Message slices.
func promptMsgsToMain(pmsgs []prompt.Message) []Message {
	out := make([]Message, len(pmsgs))
	for i, pm := range pmsgs {
		out[i] = Message{Role: pm.Role, Content: pm.Content, Name: pm.Name}
	}
	return out
}

// mainMsgsToPrompt converts main-package Message slices to prompt.Message slices.
func mainMsgsToPrompt(msgs []Message) []prompt.Message {
	out := make([]prompt.Message, len(msgs))
	for i, m := range msgs {
		out[i] = prompt.Message{Role: m.Role, Content: m.Content, Name: m.Name}
	}
	return out
}

// msgsWithFormatNudge returns a copy of the conversation with a transient
// format reminder appended. The original slice remains unmodified.
func (m *Model) msgsWithFormatNudge(conv []Message) []Message {
	nudged := m.ns.WithNudge(mainMsgsToPrompt(conv), m.modelTier())
	return promptMsgsToMain(nudged)
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
		m.help.Width = msg.Width
		m.progress.Width = msg.Width / 3
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

	case progress.FrameMsg:
		var cmd tea.Cmd
		progressModel, cmd := m.progress.Update(msg)
		m.progress = progressModel.(progress.Model)
		return m, cmd

	case msgLoadProgress:
		m.loadStage = msg.stage
		cmd := m.progress.SetPercent(msg.pct / 100.0)
		m.syncViewport()
		return m, tea.Batch(cmd, cmdReadLoadLine(m.proc))

	case msgServerReady:
		m.loadStage = ""
		plog.L.Info("model ready", "model", msg.model, "vram_mb", msg.vramMB, "native", msg.useNative)
		m.modelName = msg.model
		m.vramMB = msg.vramMB
		m.loadS = msg.loadS
		m.useNative = msg.useNative
		// Stay in stateLoading until msgAutoContext injects file context.
		// This prevents the race where the user submits before priming is done.
		m.claudeMD = loadClaudeFile(m.cfg.cwd)
		m.initNamespace()
		m.conversation = []Message{{Role: "system", Content: m.buildSystemPrompt()}}
		statusLine := fmt.Sprintf("Model loaded in %.1fs · %d MB VRAM · native=%v", msg.loadS, msg.vramMB, msg.useNative)
		if m.claudeMD != "" {
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
		return m, cmdReadLoadLine(m.proc)

	case msgServerError:
		m.selectorError = msg.err.Error()
		m.state = stateModelSelect
		m.selectorReturn = true
		m.syncViewport()
		return m, nil

	// ── Auto file context injected at startup ───────────────────────────────
	case msgAutoContext:
		fl := msg.fileList
		if fl == "(no matches)" || fl == "(no output)" {
			fl = ""
		}
		// Strip any existing primed context before injecting fresh priming.
		// Without this, repeated cd/cwd/clear accumulate duplicate stale listings.
		m.conversation = stripPrimedContext(m.conversation)
		// Rebuild system prompt with file listing embedded directly.
		// Avoids fake tool-call exchanges that small models mimic unprompted.
		if ctx, ok := m.ns.Context.(*prompt.WorkspaceContext); ok {
			ctx.Files = fl
		}
		m.conversation[0].Content = m.buildSystemPrompt()
		// Few-shot example right after system prompt so small models
		// see the expected output format before any real user messages.
		m.conversation = append(m.conversation, promptMsgsToMain(m.buildPriming())...)

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
		plog.L.Debug("inference complete", "tokens", msg.tokens, "elapsed", msg.elapsed)

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
			return m, m.proc.startInfer(m.msgsWithFormatNudge(m.conversation), 1024, 0.2)
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

		calls := m.turn().ParseCalls(reply)
		if len(calls) == 0 {
			clean := stripMarkup(reply)
			if clean == "" {
				clean = reply
			}
			rendered := renderMarkdown(clean, m.vp.Width-2)
			m.displayLines = append(m.displayLines,
				rendered,
				style.Dim.Render(fmt.Sprintf("[%d tok · %.1fs]", m.totalTokens, m.totalTime)),
				"",
			)
			m.conversation = append(m.conversation, Message{Role: "assistant", Content: reply})
			if m.cfg.review && m.critic() != nil {
				userMsg := lastUserMsg(m.conversation)
				m.displayLines = append(m.displayLines, style.Dim.Render("─── critic ───"))
				m.state = stateReview
				m.syncViewport()
				return m, m.critic().startReview(userMsg, clean)
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
		plog.L.Debug("tool done", "name", msg.name)
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
			Render(wordwrap.String(display, w-4))

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
		m.conversation = append(m.conversation, m.turn().FormatResult(msg.name, convResult))

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
		return m, m.critic().nextReviewToken()

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
				style.Dim.Render(renderMarkdown(critique, m.vp.Width-2)), "")
		}
		m.state = stateInput
		m.syncViewport()
		return m, nil

	// ── Direct shell passthrough ──────────────────────────────────────────────
	case msgShellDirect:
		m.displayLines = append(m.displayLines,
			wordwrap.String(strings.TrimRight(msg.output, "\n"), m.vp.Width-2), "")
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
				if m.selectorCursor < len(m.cfg.catalog.All())-1 {
					m.selectorCursor++
				}
			case tea.KeyEnter:
				selected := m.cfg.catalog.All()[m.selectorCursor]
				m.cfg.model = selected.Key
				m.selectorError = "" // clear any previous load error
				if m.selectorReturn {
					// In-session switch: restart sidecar with new model
					m.selectorReturn = false
					if selected.Key == m.modelName {
						m.displayLines = append(m.displayLines, style.Dim.Render("already using "+selected.Key))
						m.state = stateInput
						m.syncViewport()
						return m, nil
					}
					proc, err := m.proc.restart(selected.Key, m.cfg.projectRoot)
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
						style.Dim.Render(fmt.Sprintf("switching to %s...", selected.Key)), "")
					m.syncViewport()
					return m, cmdReadLoadLine(m.proc)
				}
				// Startup flow: launch sidecar for the first time
				m.displayLines = append(m.displayLines,
					style.Dim.Render(fmt.Sprintf("starting %s...", selected.Key)), "")
				m.syncViewport()
				return m, cmdStartSidecar(selected.Key, m.cfg.projectRoot, m.cfg.quant)
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
				return m, cmdReadLoadLine(m.proc)
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
				return m, cmdReadLoadLine(m.proc)
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
				return m, m.proc.startInfer(m.msgsWithFormatNudge(m.conversation), 1024, 0.2)

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
				m.conversation = append(m.conversation, m.turn().FormatResult("ask_user", answer))
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
	m.conversation = append(m.conversation, m.turn().FormatDenial(name))
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
	return m.proc.startInfer(m.msgsWithFormatNudge(m.conversation), 1024, 0.2)
}

// Context compaction ----------------------------------------------------------

// estimateTokens returns a rough token count for a conversation.
// Uses ~4 chars per token as a conservative estimate for English text.
func estimateTokens(conv []Message) int {
	total := 0
	for _, msg := range conv {
		total += len(msg.Content)/4 + 4 // +4 for role/framing overhead per message
	}
	return total
}

// compactIfNeeded trims the conversation when it exceeds the message count
// threshold OR the model's context token limit. Returns removed messages
// (nil if no compaction was needed).
func (m *Model) compactIfNeeded() []Message {
	needsCompact := len(m.conversation) > compactThreshold

	// Also compact if estimated tokens approach the model's context limit.
	limit := m.cfg.catalog.ContextLimit(m.modelName)
	if estimateTokens(m.conversation) > limit-tokenReserve {
		needsCompact = true
	}

	if !needsCompact {
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

// stripPrimedContext removes any previously injected few-shot priming messages
// from the conversation, identified by known sentinel user prompts.
func stripPrimedContext(conv []Message) []Message {
	sentinels := map[string]bool{
		"What causes stress?":                                       true, // Tier 1
		"How does attachment style affect adult relationships?":      true, // Tier 2/3
	}
	out := make([]Message, 0, len(conv))
	skip := false
	for _, msg := range conv {
		if msg.Role == "user" && sentinels[msg.Content] {
			skip = true // skip this user msg and the following assistant reply
			continue
		}
		if skip {
			skip = false // consumed the assistant reply
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
		m.claudeMD = loadClaudeFile(m.cfg.cwd)
		m.initNamespace()
		m.conversation = []Message{{Role: "system", Content: m.buildSystemPrompt()}}
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
			for i, mi := range m.cfg.catalog.All() {
				if mi.Key == m.modelName {
					m.selectorCursor = i
					break
				}
			}
			m.state = stateModelSelect
			m.input.Blur()
			return nil
		}
		newModel := parts[1]
		if m.cfg.catalog.Get(newModel) == nil {
			m.displayLines = append(m.displayLines, style.Error.Render(
				fmt.Sprintf("unknown model %q — valid: %s", newModel, strings.Join(m.cfg.catalog.ValidKeys(), ", "))))
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
		return cmdReadLoadLine(m.proc)

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
		path, err := session.ExportMarkdown(m.modelName, sessionMsgsFromMain(m.conversation), m.cfg.cwd)
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
			style.Dim.Render("models: "+strings.Join(m.cfg.catalog.ValidKeys(), "  ")),
			style.Dim.Render("tools needing approval: shell, write_file  (bypass: --yes)"),
			style.Dim.Render("--budget N  — stop after N total tokens (0 = unlimited)"),
			style.Dim.Render("--review    — enable adversarial critic pass after each response"),
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
		path, err := session.Save(m.modelName, sessionMsgsFromMain(m.conversation))
		if err != nil {
			m.displayLines = append(m.displayLines, style.Error.Render("save failed: "+err.Error()))
		} else {
			m.displayLines = append(m.displayLines, style.Dim.Render("saved: "+path))
		}
	case "list":
		sessions, err := session.List()
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
		s, err := session.LoadByIndex(n)
		if err != nil {
			m.displayLines = append(m.displayLines, style.Error.Render("session not found"))
			return
		}
		m.resetToolState()
		m.conversation = sessionMsgsToMain(s.Conversation)
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

// rebuildSystemPrompt regenerates conversation[0] from the namespace.
// Call whenever skills, cwd, or claudeMD change.
func (m *Model) rebuildSystemPrompt() {
	m.initNamespace()
	base := m.buildSystemPrompt()
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

// turn returns a Turn configured for the current model's format mode.
func (m *Model) turn() Turn {
	return Turn{Native: m.useNative, CWD: m.cfg.cwd}
}

// critic returns the reviewer interface from the current proc.
// Returns nil if the proc does not support critic review.
func (m *Model) critic() reviewer {
	if r, ok := m.proc.(reviewer); ok {
		return r
	}
	return nil
}

// renderMarkdown renders markdown text to styled terminal output at the given width.
// Falls back to plain word-wrapped text on render errors.
func renderMarkdown(text string, width int) string {
	if width < 10 {
		width = 10
	}
	r, err := glamour.NewTermRenderer(
		glamour.WithStandardStyle("dark"),
		glamour.WithWordWrap(width),
	)
	if err != nil {
		return wordwrap.String(text, width)
	}
	out, err := r.Render(text)
	if err != nil {
		return wordwrap.String(text, width)
	}
	return strings.TrimRight(out, "\n")
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
		wrapped := wordwrap.String(string(m.streamBuf), m.vp.Width-2)
		content += style.Value.Render(wrapped)
	}
	if len(m.reviewBuf) > 0 {
		if content != "" {
			content += "\n"
		}
		wrapped := wordwrap.String(string(m.reviewBuf), m.vp.Width-2)
		content += style.Dim.Render(wrapped)
	}
	m.vp.SetContent(content)
	if wasAtBottom || activeStream {
		m.vp.GotoBottom()
	}
}

func (m Model) chatHeight() int {
	// header(1) + statusLine(1) + helpBar(1) + inputArea + border padding
	chrome := 4
	switch m.state {
	case stateInput, stateAskUser:
		chrome += m.input.Height() // textarea visible lines
	case stateConfirm:
		chrome += 2 // tool detail + y/N prompt
	default:
		chrome += 1
	}
	h := m.height - chrome
	if h < 4 {
		h = 4
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
		if m.loadStage != "" {
			statusLine = m.spinner.View() + " " + m.progress.View() + " " + style.Dim.Render(m.loadStage)
		} else {
			statusLine = m.spinner.View() + " " + style.Dim.Render("loading model...")
		}
	case stateThinking:
		statusLine = m.spinner.View() + " " + style.Dim.Render("thinking...")
	case stateReview:
		statusLine = m.spinner.View() + " " + style.Dim.Render("critic reviewing...")
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

	helpBar := m.help.View(helpKeyMap{m.stateHelp()})

	var inputBar string
	switch m.state {
	case stateInput, stateAskUser:
		inputBar = m.input.View()
	case stateConfirm:
		if len(m.pendingTools) > m.toolIdx {
			tool := m.pendingTools[m.toolIdx]
			argsJSON, _ := json.Marshal(tool.Args)
			inputBar = style.Warning.Render("["+tool.Name+"] ") + style.Dim.Render(string(argsJSON)) + "\n" +
				style.Heading.Render("Run this tool? [y/N] ")
		}
	}

	parts := []string{header, m.vp.View(), statusLine, helpBar}
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
	errBanner := ""
	if m.selectorError != "" {
		errBanner = "\n" + style.Error.Render("  "+m.selectorError) + "\n"
	}
	content := m.renderModelPicker(m.height - 4)
	hint := style.Dim.Render("  ↑/↓ navigate · Enter select · Esc " +
		func() string {
			if m.selectorReturn {
				return "cancel"
			}
			return "quit"
		}())
	return header + errBanner + "\n" + content + "\n" + hint
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

	// Column header
	b.WriteString(dimStyle.Render("     Model                Tier    VRAM     Context") + "\n")

	for i, mi := range m.cfg.catalog.All() {
		cursor := "  "
		nameStyle := dimStyle
		if i == m.selectorCursor {
			cursor = cursorStyle.Render("> ")
			nameStyle = selectedStyle
		}

		note := mi.DisplayNote()
		line := fmt.Sprintf("%s%-20s  %-7s %-8s %-5s",
			cursor,
			nameStyle.Render(mi.Label),
			dimStyle.Render(mi.TierLabel()),
			dimStyle.Render(mi.VRAM),
			dimStyle.Render(mi.DeclaredCtx),
		)
		if note != "" {
			line += "  " + dimStyle.Render(note)
		}
		b.WriteString(line + "\n")

		if i >= maxHeight-5 {
			break
		}
	}
	b.WriteString("\n")
	return b.String()
}

