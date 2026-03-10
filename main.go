// Package main implements pai: a Socratic psychology agent interface backed by local LLMs.
// The LLM inference runs in a persistent Python sidecar (scripts/llm-serve.py)
// that keeps the model in VRAM between turns, or via an HTTP gateway.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/catalog"
	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/config"
	plog "github.com/safety-quotient-lab/psychology-agent-interface/pkg/log"
	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/prompt"
	"github.com/spf13/cobra"
)

// inferProc is the backend inference interface. Two implementations:
//   - *llmProc  — Python sidecar via JSON-lines stdio (local, with streaming)
//   - *httpProc — HTTP POST to llm-gateway or llm-infer-http (remote, non-streaming)
type inferProc interface {
	waitReady() (readyPayload, error)
	// infer runs one blocking inference turn. Used by --print mode.
	infer(msgs []Message, maxTokens int, temp float64) (inferResponse, error)
	startInfer(msgs []Message, maxTokens int, temp float64) tea.Cmd
	nextToken() tea.Cmd
	startReview(userMsg, primaryReply string) tea.Cmd
	nextReviewToken() tea.Cmd
	// restart kills the current backend and returns a fresh one for the given model.
	// HTTP backends return themselves (gateway is always up).
	restart(model, projectRoot string) (inferProc, error)
}

// ─────────────────────────────────────────────────────────────
// llmProc — subprocess / JSON-lines sidecar
// ─────────────────────────────────────────────────────────────

// llmProc wraps the Python sidecar and its stdio pipes.
type llmProc struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	stderr *bytes.Buffer // captured stderr for error reporting
}

// startSidecar starts scripts/llm-serve.py and creates JSON-lines pipes.
func startSidecar(model, projectRoot string) (*llmProc, error) {
	return startSidecarQuant(model, projectRoot, "")
}

func startSidecarQuant(model, projectRoot, quant string) (*llmProc, error) {
	script := filepath.Join(projectRoot, "scripts", "llm-serve.py")
	args := []string{script, "--model", model}
	if quant != "" {
		args = append(args, "--quant", quant)
	}
	cmd := exec.Command("python3", args...)
	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start sidecar: %w", err)
	}

	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 1<<20), 1<<20) // 1 MB — handles large replies

	return &llmProc{cmd: cmd, stdin: stdin, stdout: sc, stderr: &stderrBuf}, nil
}

// readReady reads the startup handshake from the sidecar (blocking).
// Skips any progress lines emitted before the ready payload.
func (p *llmProc) readReady() (readyPayload, error) {
	for p.stdout.Scan() {
		var msg readyPayload
		if err := json.Unmarshal(p.stdout.Bytes(), &msg); err != nil {
			return readyPayload{}, fmt.Errorf("parse ready handshake: %w", err)
		}
		if msg.Status == "ready" {
			return msg, nil
		}
		// Progress line — skip and read next
	}
	if err := p.stdout.Err(); err != nil {
		return readyPayload{}, fmt.Errorf("sidecar read: %w", err)
	}
	return readyPayload{}, fmt.Errorf("sidecar closed before ready: %s", p.lastStderrLine())
}

// lastStderrLine returns the last non-empty line from captured stderr,
// providing a concise error summary when the sidecar crashes.
func (p *llmProc) lastStderrLine() string {
	lines := strings.Split(strings.TrimSpace(p.stderr.String()), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if line != "" {
			return line
		}
	}
	return "(no stderr output)"
}

// sendInfer sends one inference request and reads the response (blocking).
func (p *llmProc) sendInfer(req inferRequest) (inferResponse, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return inferResponse{}, err
	}
	if _, err := fmt.Fprintf(p.stdin, "%s\n", b); err != nil {
		return inferResponse{}, fmt.Errorf("write to sidecar: %w", err)
	}
	if !p.stdout.Scan() {
		if err := p.stdout.Err(); err != nil {
			return inferResponse{}, fmt.Errorf("sidecar read: %w", err)
		}
		return inferResponse{}, fmt.Errorf("sidecar closed")
	}
	var resp inferResponse
	if err := json.Unmarshal(p.stdout.Bytes(), &resp); err != nil {
		return inferResponse{}, fmt.Errorf("parse response: %w", err)
	}
	return resp, nil
}

// inferProc interface implementation for llmProc.

func (p *llmProc) waitReady() (readyPayload, error) { return p.readReady() }

func (p *llmProc) infer(msgs []Message, maxTokens int, temp float64) (inferResponse, error) {
	return p.sendInfer(inferRequest{Messages: msgs, MaxTokens: maxTokens, Temperature: temp, Stream: false})
}

// inferStream sends a streaming inference request, writing each token to w as
// it arrives. Returns the full concatenated reply and final token count.
func (p *llmProc) inferStream(msgs []Message, maxTokens int, temp float64, w io.Writer) (string, int, error) {
	b, err := json.Marshal(inferRequest{Messages: msgs, MaxTokens: maxTokens, Temperature: temp, Stream: true})
	if err != nil {
		return "", 0, err
	}
	if _, err := fmt.Fprintf(p.stdin, "%s\n", b); err != nil {
		return "", 0, fmt.Errorf("write to sidecar: %w", err)
	}
	var sb strings.Builder
	var totalTokens int
	for p.stdout.Scan() {
		line := p.stdout.Bytes()
		var msg struct {
			Token   string `json:"token"`
			Done    bool   `json:"done"`
			Tokens  int    `json:"tokens"`
			Error   string `json:"error"`
		}
		if err := json.Unmarshal(line, &msg); err != nil {
			continue
		}
		if msg.Error != "" {
			return sb.String(), totalTokens, fmt.Errorf("sidecar: %s", msg.Error)
		}
		if msg.Done {
			totalTokens = msg.Tokens
			break
		}
		if msg.Token != "" {
			sb.WriteString(msg.Token)
			fmt.Fprint(w, msg.Token)
		}
	}
	if err := p.stdout.Err(); err != nil {
		return sb.String(), totalTokens, err
	}
	return sb.String(), totalTokens, nil
}

func (p *llmProc) startInfer(msgs []Message, maxTokens int, temp float64) tea.Cmd {
	return func() tea.Msg {
		b, err := json.Marshal(inferRequest{Messages: msgs, MaxTokens: maxTokens, Temperature: temp, Stream: true})
		if err != nil {
			return msgStreamDone{err: err}
		}
		if _, err := fmt.Fprintf(p.stdin, "%s\n", b); err != nil {
			return msgStreamDone{err: err}
		}
		return readStreamLine(p)
	}
}

func (p *llmProc) nextToken() tea.Cmd {
	return func() tea.Msg { return readStreamLine(p) }
}

func (p *llmProc) startReview(userMsg, primaryReply string) tea.Cmd {
	msgs := []Message{{Role: "user", Content: prompt.CriticPrompt(userMsg, primaryReply)}}
	return func() tea.Msg {
		b, err := json.Marshal(inferRequest{Messages: msgs, MaxTokens: 512, Temperature: 0.1, Stream: true})
		if err != nil {
			return msgReviewDone{err: err}
		}
		if _, err := fmt.Fprintf(p.stdin, "%s\n", b); err != nil {
			return msgReviewDone{err: err}
		}
		return readReviewLine(p)
	}
}

func (p *llmProc) nextReviewToken() tea.Cmd {
	return func() tea.Msg { return readReviewLine(p) }
}

func (p *llmProc) restart(model, projectRoot string) (inferProc, error) {
	p.cmd.Process.Kill() //nolint:errcheck
	return startSidecar(model, projectRoot)
}

// ─────────────────────────────────────────────────────────────
// httpProc — HTTP gateway backend (no streaming)
// ─────────────────────────────────────────────────────────────

type httpProc struct {
	baseURL    string
	model      string
	sshTunnel  *exec.Cmd // non-nil when we own the SSH tunnel
	client     *http.Client
	streamBody io.ReadCloser  // held open during active HTTP streaming
	streamSc   *bufio.Scanner // scanner over streamBody
}

// httpStreamLine is one newline-delimited JSON chunk from a streaming /infer response.
type httpStreamLine struct {
	Token   string  `json:"token"`
	Done    bool    `json:"done"`
	Tokens  int     `json:"tokens"`
	Seconds float64 `json:"seconds"`
	Error   string  `json:"error"`
}

func (p *httpProc) closeStream() {
	if p.streamBody != nil {
		p.streamBody.Close()
		p.streamBody = nil
		p.streamSc = nil
	}
}

func (p *httpProc) readHTTPStreamLine(isReview bool) tea.Msg {
	if p.streamSc == nil || !p.streamSc.Scan() {
		p.closeStream()
		if isReview {
			return msgReviewDone{}
		}
		return msgStreamDone{}
	}
	var item httpStreamLine
	if err := json.Unmarshal(p.streamSc.Bytes(), &item); err != nil {
		p.closeStream()
		if isReview {
			return msgReviewDone{err: err}
		}
		return msgStreamDone{err: err}
	}
	if item.Error != "" {
		p.closeStream()
		e := fmt.Errorf("%s", item.Error)
		if isReview {
			return msgReviewDone{err: e}
		}
		return msgStreamDone{err: e}
	}
	if item.Done {
		p.closeStream()
		if isReview {
			return msgReviewDone{tokens: item.Tokens, elapsed: item.Seconds}
		}
		return msgStreamDone{tokens: item.Tokens, elapsed: item.Seconds}
	}
	if isReview {
		return msgReviewToken{token: item.Token}
	}
	return msgToken{token: item.Token}
}

func (p *httpProc) waitReady() (readyPayload, error) {
	// Probe GET /health — retry for up to 15s (tunnel may be coming up)
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := p.client.Get(p.baseURL + "/health")
		if err == nil {
			defer resp.Body.Close()
			var h struct {
				Status string `json:"status"`
			}
			json.NewDecoder(resp.Body).Decode(&h) //nolint:errcheck
			if h.Status == "ready" || h.Status == "degraded" {
				return readyPayload{
					Status:    "ready",
					Model:     p.model,
					UseNative: strings.HasPrefix(p.model, "qwen"),
				}, nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return readyPayload{}, fmt.Errorf("gateway not ready after 15s")
}

func (p *httpProc) doInfer(msgs []Message, maxTokens int, temp float64) (inferResponse, error) {
	body, err := json.Marshal(map[string]any{
		"messages":    msgs,
		"model":       p.model,
		"max_tokens":  maxTokens,
		"temperature": temp,
	})
	if err != nil {
		return inferResponse{}, err
	}
	resp, err := p.client.Post(p.baseURL+"/infer", "application/json", bytes.NewReader(body))
	if err != nil {
		return inferResponse{}, err
	}
	defer resp.Body.Close()
	var ir inferResponse
	if err := json.NewDecoder(resp.Body).Decode(&ir); err != nil {
		return inferResponse{}, err
	}
	if ir.Error != "" {
		return inferResponse{}, fmt.Errorf("%s", ir.Error)
	}
	return ir, nil
}

func (p *httpProc) openStream(msgs []Message, maxTokens int, temp float64) error {
	body, err := json.Marshal(map[string]any{
		"messages":    msgs,
		"model":       p.model,
		"max_tokens":  maxTokens,
		"temperature": temp,
		"stream":      true,
	})
	if err != nil {
		return err
	}
	resp, err := p.client.Post(p.baseURL+"/infer", "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	p.streamBody = resp.Body
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 256*1024), 256*1024)
	p.streamSc = sc
	return nil
}

func (p *httpProc) startInfer(msgs []Message, maxTokens int, temp float64) tea.Cmd {
	return func() tea.Msg {
		if err := p.openStream(msgs, maxTokens, temp); err != nil {
			return msgStreamDone{err: err}
		}
		return p.readHTTPStreamLine(false)
	}
}

func (p *httpProc) infer(msgs []Message, maxTokens int, temp float64) (inferResponse, error) {
	return p.doInfer(msgs, maxTokens, temp)
}

func (p *httpProc) nextToken() tea.Cmd {
	return func() tea.Msg { return p.readHTTPStreamLine(false) }
}

func (p *httpProc) startReview(userMsg, primaryReply string) tea.Cmd {
	msgs := []Message{{Role: "user", Content: prompt.CriticPrompt(userMsg, primaryReply)}}
	return func() tea.Msg {
		if err := p.openStream(msgs, 512, 0.1); err != nil {
			return msgReviewDone{err: err}
		}
		return p.readHTTPStreamLine(true)
	}
}

func (p *httpProc) nextReviewToken() tea.Cmd {
	return func() tea.Msg { return p.readHTTPStreamLine(true) }
}

func (p *httpProc) restart(model, _ string) (inferProc, error) {
	p.model = model // gateway supports any model — just update the field
	return p, nil
}

// startSSHTunnel opens ssh -N -L localPort:localhost:remotePort host and waits
// until the local port is connectable (up to 10s). Returns the ssh Cmd for cleanup.
func startSSHTunnel(sshHost string, localPort, remotePort int) (*exec.Cmd, error) {
	tunnel := fmt.Sprintf("%d:localhost:%d", localPort, remotePort)
	cmd := exec.Command("ssh", "-N", "-o", "StrictHostKeyChecking=no",
		"-o", "ExitOnForwardFailure=yes",
		"-L", tunnel, sshHost)
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("ssh tunnel start: %w", err)
	}
	// Wait until port is connectable
	addr := fmt.Sprintf("127.0.0.1:%d", localPort)
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 500*time.Millisecond)
		if err == nil {
			conn.Close()
			return cmd, nil
		}
		time.Sleep(200 * time.Millisecond)
	}
	cmd.Process.Kill() //nolint:errcheck
	return nil, fmt.Errorf("ssh tunnel to %s not connectable within 10s", sshHost)
}

// Wire types for the JSON-lines protocol.

type readyPayload struct {
	Status    string  `json:"status"`
	Model     string  `json:"model"`
	VramMB    int     `json:"vram_mb"`
	LoadS     float64 `json:"load_s"`
	UseNative bool    `json:"use_native"`
}

type inferRequest struct {
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature float64   `json:"temperature"`
	Stream      bool      `json:"stream"`
}

type inferResponse struct {
	Reply   string  `json:"reply"`
	Tokens  int     `json:"tokens"`
	Seconds float64 `json:"seconds"`
	Error   string  `json:"error"`
}

// appConfig holds resolved CLI flags.
type appConfig struct {
	model       string
	cwd         string
	maxTurns    int
	autoApprove bool
	projectRoot string
	tokenBudget int
	review      bool             // enable adversarial critic pass after each response (off by default)
	quant       string           // quantization: "4bit", "8bit", or "" (fp16)
	gateway     string           // SSH host for LLM gateway (e.g. "mac-mini"); empty = subprocess
	print       bool             // headless print mode — no TUI, prompt from args or stdin
	stream      bool             // stream tokens to stdout in print mode
	prompt      string           // joined positional args (print mode only)
	modelSet    bool             // true if --model was explicitly provided
	catalog     *catalog.Catalog // model metadata loaded from models.json
}

func main() {
	var c appConfig
	root := &cobra.Command{
		Use:   "pai [prompt...]",
		Short: "Socratic psychology agent interface with local LLM",
		Args:  cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) > 0 {
				c.prompt = strings.Join(args, " ")
				c.print = true
			}
			c.modelSet = cmd.Flags().Changed("model")
			return run(c)
		},
	}
	root.Flags().StringVarP(&c.model, "model", "m", "qwen-0.5b",
		"Model key (qwen-0.5b, qwen-1.5b, qwen-3b, smollm2, gemma-2b, llama-1b, llama-3b)")
	root.Flags().StringVarP(&c.cwd, "cwd", "C", mustGetwd(),
		"Working directory for tools")
	root.Flags().IntVarP(&c.maxTurns, "max-turns", "n", 15,
		"Tool-call limit per exchange")
	root.Flags().BoolVarP(&c.autoApprove, "yes", "y", false,
		"Auto-approve all tool calls without prompting")
	root.Flags().IntVar(&c.tokenBudget, "budget", 0,
		"Total token budget for session (0 = unlimited)")
	root.Flags().BoolVar(&c.review, "review", false,
		"Enable adversarial critic pass after each response (off by default)")
	root.Flags().StringVar(&c.quant, "quant", "",
		"Quantization: 4bit (~1.7GB for 3b) or 8bit (~3.5GB for 3b), or empty for fp16 (CUDA only)")
	root.Flags().StringVar(&c.gateway, "gateway", "",
		"SSH host for LLM gateway on port 7705 (e.g. mac-mini). Tunnels :7705 locally.")
	root.Flags().BoolVarP(&c.print, "print", "p", false,
		"Headless print mode — run one prompt to completion and print reply to stdout")
	root.Flags().BoolVarP(&c.stream, "stream", "s", false,
		"Stream tokens to stdout as they are generated (print mode only)")

	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

func run(c appConfig) error {
	plog.L.Info("pai starting", "model", c.model, "cwd", c.cwd)
	// Load .dev.vars from home → cwd chain; closer files override outer ones.
	home, _ := os.UserHomeDir()
	var devVarsDirs []string
	for dir := c.cwd; ; dir = filepath.Dir(dir) {
		devVarsDirs = append(devVarsDirs, dir)
		if dir == home || filepath.Dir(dir) == dir {
			break
		}
	}
	for i := len(devVarsDirs) - 1; i >= 0; i-- {
		config.LoadDotFile(filepath.Join(devVarsDirs[i], ".dev.vars")) //nolint:errcheck
	}

	c.projectRoot = findProjectRoot()

	// Load model catalog from models.json.
	cat, err := catalog.Load(c.projectRoot)
	if err != nil {
		return fmt.Errorf("model catalog: %w", err)
	}
	c.catalog = cat

	// Auto-detect print mode: explicit --print flag, positional args, or piped stdin.
	if !c.print {
		stat, _ := os.Stdin.Stat()
		if (stat.Mode() & os.ModeCharDevice) == 0 {
			c.print = true
		}
	}

	// Interactive model selector: when --model not explicitly set and in TUI mode,
	// launch the TUI with the model picker before starting the sidecar.
	if !c.modelSet && !c.print {
		p := tea.NewProgram(newModelSelector(c), tea.WithAltScreen(), tea.WithMouseCellMotion())
		finalModel, err := p.Run()
		if err != nil {
			return err
		}
		if m, ok := finalModel.(Model); ok && len(m.conversation) > 1 {
			jsonlPath, jerr := exportJSONL(m.modelName, m.conversation, c.cwd)
			if jerr == nil {
				htmlPath := strings.TrimSuffix(jsonlPath, ".jsonl") + ".html"
				title := fmt.Sprintf("pai [%s] — %s", m.modelName, strings.TrimSuffix(filepath.Base(jsonlPath), ".jsonl"))
				if rerr := generateReplay(jsonlPath, htmlPath, title); rerr == nil {
					fmt.Fprintf(os.Stderr, "\n🎬  session replay: file://%s\n", htmlPath)
				}
			}
		}
		return err
	}

	// Validate model key before touching Python (required when --model set explicitly).
	if c.catalog.Get(c.model) == nil {
		return fmt.Errorf("unknown model %q — valid choices: %s", c.model, strings.Join(c.catalog.ValidKeys(), ", "))
	}

	var proc inferProc
	if c.gateway != "" {
		fmt.Fprintf(os.Stderr, "Tunneling gateway port 7705 via %s...\n", c.gateway)
		tunnel, err := startSSHTunnel(c.gateway, 7705, 7705)
		if err != nil {
			return fmt.Errorf("SSH tunnel to %s: %w", c.gateway, err)
		}
		defer tunnel.Process.Kill() //nolint:errcheck
		plog.L.Info("ssh tunnel opened", "host", c.gateway)
		proc = &httpProc{
			baseURL: "http://127.0.0.1:7705",
			model:   c.model,
			sshTunnel: tunnel,
			client:  &http.Client{Timeout: 300 * time.Second},
		}
	} else {
		sp, err := startSidecarQuant(c.model, c.projectRoot, c.quant)
		if err != nil {
			return fmt.Errorf("failed to start llm-serve.py: %w", err)
		}
		defer sp.cmd.Process.Kill() //nolint:errcheck
		plog.L.Info("sidecar started", "model", c.model)
		proc = sp
	}

	if c.print {
		return runPrint(c, proc)
	}

	p := tea.NewProgram(newModel(c, proc), tea.WithAltScreen(), tea.WithMouseCellMotion())
	finalModel, err := p.Run()

	// Generate a session replay after the TUI exits.
	if m, ok := finalModel.(Model); ok && len(m.conversation) > 1 {
		jsonlPath, jerr := exportJSONL(m.modelName, m.conversation, c.cwd)
		if jerr == nil {
			htmlPath := strings.TrimSuffix(jsonlPath, ".jsonl") + ".html"
			title := fmt.Sprintf("pai [%s] — %s", m.modelName, strings.TrimSuffix(filepath.Base(jsonlPath), ".jsonl"))
			if rerr := generateReplay(jsonlPath, htmlPath, title); rerr == nil {
				fmt.Fprintf(os.Stderr, "\n🎬  session replay: file://%s\n", htmlPath)
			}
		}
	}

	return err
}

func mustGetwd() string {
	d, _ := os.Getwd()
	return d
}

func findProjectRoot() string {
	// Prefer the directory containing the executable — works when pai is on PATH.
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		if _, err := os.Stat(filepath.Join(dir, "scripts", "llm-serve.py")); err == nil {
			return dir
		}
	}
	// Fallback: walk up from cwd looking for go.mod (dev/source workflow).
	dir, _ := os.Getwd()
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	d, _ := os.Getwd()
	return d
}
