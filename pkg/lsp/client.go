// Package lsp provides a JSON-RPC client for gopls (Go language server).
// Manages the gopls subprocess lifecycle and exposes LSP operations
// (definition, references, hover, diagnostics) for use as agent tools.
package lsp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
)

// Client manages a gopls subprocess communicating via JSON-RPC over stdin/stdout.
type Client struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	mu     sync.Mutex
	nextID atomic.Int64
	root   string // workspace root (absolute path)
}

// jsonRPCRequest represents a JSON-RPC 2.0 request.
type jsonRPCRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// jsonRPCResponse represents a JSON-RPC 2.0 response.
type jsonRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// NewClient starts gopls for the given workspace root and performs the
// LSP initialize handshake. Returns an error if gopls cannot start or
// the handshake fails.
func NewClient(workspaceRoot string) (*Client, error) {
	goplsPath := findGopls()
	if goplsPath == "" {
		return nil, fmt.Errorf("gopls not found — install with: go install golang.org/x/tools/gopls@latest")
	}

	absRoot, err := filepath.Abs(workspaceRoot)
	if err != nil {
		return nil, fmt.Errorf("resolve workspace root: %w", err)
	}

	cmd := exec.Command(goplsPath, "serve")
	cmd.Stderr = io.Discard

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("stdin pipe: %w", err)
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start gopls: %w", err)
	}

	c := &Client{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewReader(stdoutPipe),
		root:   absRoot,
	}

	if err := c.initialize(); err != nil {
		c.Close()
		return nil, fmt.Errorf("LSP initialize: %w", err)
	}

	return c, nil
}

// Close shuts down the gopls subprocess gracefully.
func (c *Client) Close() error {
	// Send shutdown request, ignore errors
	_, _ = c.call("shutdown", nil)
	// Send exit notification (no response expected)
	_ = c.notify("exit", nil)
	_ = c.stdin.Close()
	return c.cmd.Wait()
}

// Definition returns the definition location for the symbol at the given
// file position. Returns a human-readable "file:line:col" string.
func (c *Client) Definition(file string, line, col int) (string, error) {
	file = c.absPath(file)
	c.didOpen(file)
	defer c.didClose(file)

	params := textDocumentPositionParams(file, line, col)
	raw, err := c.call("textDocument/definition", params)
	if err != nil {
		return "", err
	}
	return formatLocations(raw, c.root)
}

// References returns all references to the symbol at the given position.
func (c *Client) References(file string, line, col int) (string, error) {
	file = c.absPath(file)
	c.didOpen(file)
	defer c.didClose(file)

	params := map[string]any{
		"textDocument": map[string]string{"uri": fileURI(file)},
		"position":     position(line, col),
		"context":      map[string]bool{"includeDeclaration": true},
	}
	raw, err := c.call("textDocument/references", params)
	if err != nil {
		return "", err
	}
	return formatLocations(raw, c.root)
}

// Hover returns type/doc information for the symbol at the given position.
func (c *Client) Hover(file string, line, col int) (string, error) {
	file = c.absPath(file)
	c.didOpen(file)
	defer c.didClose(file)

	params := textDocumentPositionParams(file, line, col)
	raw, err := c.call("textDocument/hover", params)
	if err != nil {
		return "", err
	}
	if raw == nil || string(raw) == "null" {
		return "(no hover info)", nil
	}
	var hover struct {
		Contents struct {
			Kind  string `json:"kind"`
			Value string `json:"value"`
		} `json:"contents"`
	}
	if err := json.Unmarshal(raw, &hover); err != nil {
		return "", fmt.Errorf("parse hover: %w", err)
	}
	return hover.Contents.Value, nil
}

// Diagnostics returns compiler errors and warnings for the given file.
func (c *Client) Diagnostics(file string) (string, error) {
	file = c.absPath(file)
	c.didOpen(file)
	defer c.didClose(file)

	// gopls publishes diagnostics via notifications, but we can request them
	// via textDocument/diagnostic (LSP 3.17+).
	params := map[string]any{
		"textDocument": map[string]string{"uri": fileURI(file)},
	}
	raw, err := c.call("textDocument/diagnostic", params)
	if err != nil {
		// Fallback: some gopls versions don't support pull diagnostics.
		// Return a hint rather than failing.
		return "(diagnostics not available via pull — run 'go build' for errors)", nil
	}
	if raw == nil || string(raw) == "null" {
		return "(no diagnostics)", nil
	}
	var report struct {
		Items []diagnostic `json:"items"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		return "", fmt.Errorf("parse diagnostics: %w", err)
	}
	if len(report.Items) == 0 {
		return "(no diagnostics)", nil
	}
	return formatDiagnostics(report.Items), nil
}

// --- internal ---

func (c *Client) absPath(file string) string {
	if !filepath.IsAbs(file) {
		return filepath.Join(c.root, file)
	}
	return file
}

func (c *Client) initialize() error {
	params := map[string]any{
		"processId":    os.Getpid(),
		"rootUri":      fileURI(c.root),
		"capabilities": map[string]any{},
	}
	_, err := c.call("initialize", params)
	if err != nil {
		return err
	}
	return c.notify("initialized", map[string]any{})
}

func (c *Client) didOpen(file string) {
	content, err := os.ReadFile(file)
	if err != nil {
		return
	}
	_ = c.notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{
			"uri":        fileURI(file),
			"languageId": "go",
			"version":    1,
			"text":       string(content),
		},
	})
}

func (c *Client) didClose(file string) {
	_ = c.notify("textDocument/didClose", map[string]any{
		"textDocument": map[string]string{"uri": fileURI(file)},
	})
}

func (c *Client) call(method string, params any) (json.RawMessage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	id := c.nextID.Add(1)
	req := jsonRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}
	if err := c.send(req); err != nil {
		return nil, fmt.Errorf("send %s: %w", method, err)
	}

	// Read responses, skipping notifications/requests from server
	for {
		raw, err := c.readMessage()
		if err != nil {
			return nil, fmt.Errorf("read %s response: %w", method, err)
		}
		var resp jsonRPCResponse
		if err := json.Unmarshal(raw, &resp); err != nil {
			continue // skip malformed
		}
		if resp.ID != id {
			continue // skip notifications or out-of-order responses
		}
		if resp.Error != nil {
			return nil, fmt.Errorf("gopls %s error %d: %s", method, resp.Error.Code, resp.Error.Message)
		}
		return resp.Result, nil
	}
}

func (c *Client) notify(method string, params any) error {
	msg := struct {
		JSONRPC string `json:"jsonrpc"`
		Method  string `json:"method"`
		Params  any    `json:"params,omitempty"`
	}{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}
	return c.send(msg)
}

func (c *Client) send(msg any) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(data))
	_, err = fmt.Fprint(c.stdin, header)
	if err != nil {
		return err
	}
	_, err = c.stdin.Write(data)
	return err
}

func (c *Client) readMessage() ([]byte, error) {
	// Read headers
	contentLength := 0
	for {
		line, err := c.stdout.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break // end of headers
		}
		if strings.HasPrefix(line, "Content-Length:") {
			_, _ = fmt.Sscanf(line, "Content-Length: %d", &contentLength)
		}
	}
	if contentLength == 0 {
		return nil, fmt.Errorf("missing Content-Length header")
	}
	body := make([]byte, contentLength)
	_, err := io.ReadFull(c.stdout, body)
	return body, err
}

// --- helpers ---

func findGopls() string {
	// Check PATH first
	if p, err := exec.LookPath("gopls"); err == nil {
		return p
	}
	// Check GOPATH/bin
	gopath := os.Getenv("GOPATH")
	if gopath == "" {
		home, _ := os.UserHomeDir()
		gopath = filepath.Join(home, "go")
	}
	candidate := filepath.Join(gopath, "bin", "gopls")
	if _, err := os.Stat(candidate); err == nil {
		return candidate
	}
	return ""
}

func fileURI(path string) string {
	return "file://" + path
}

func position(line, col int) map[string]int {
	// LSP uses 0-based line/col; our tools accept 1-based (human-friendly)
	return map[string]int{
		"line":      line - 1,
		"character": col - 1,
	}
}

func textDocumentPositionParams(file string, line, col int) map[string]any {
	return map[string]any{
		"textDocument": map[string]string{"uri": fileURI(file)},
		"position":     position(line, col),
	}
}

type location struct {
	URI   string `json:"uri"`
	Range struct {
		Start struct {
			Line      int `json:"line"`
			Character int `json:"character"`
		} `json:"start"`
	} `json:"range"`
}

type diagnostic struct {
	Range struct {
		Start struct {
			Line      int `json:"line"`
			Character int `json:"character"`
		} `json:"start"`
	} `json:"range"`
	Severity int    `json:"severity"`
	Message  string `json:"message"`
	Source   string `json:"source"`
}

func formatLocations(raw json.RawMessage, root string) (string, error) {
	if raw == nil || string(raw) == "null" {
		return "(no results)", nil
	}
	var locs []location
	if err := json.Unmarshal(raw, &locs); err != nil {
		// Try single location
		var single location
		if err2 := json.Unmarshal(raw, &single); err2 != nil {
			return "", fmt.Errorf("parse locations: %w", err)
		}
		locs = []location{single}
	}
	if len(locs) == 0 {
		return "(no results)", nil
	}
	var sb strings.Builder
	for _, loc := range locs {
		path := strings.TrimPrefix(loc.URI, "file://")
		// Show relative path when inside workspace
		if rel, err := filepath.Rel(root, path); err == nil && !strings.HasPrefix(rel, "..") {
			path = rel
		}
		// Convert back to 1-based for display
		fmt.Fprintf(&sb, "%s:%d:%d\n", path, loc.Range.Start.Line+1, loc.Range.Start.Character+1)
	}
	return strings.TrimRight(sb.String(), "\n"), nil
}

func formatDiagnostics(diags []diagnostic) string {
	sevName := map[int]string{1: "error", 2: "warning", 3: "info", 4: "hint"}
	var sb strings.Builder
	for _, d := range diags {
		sev := sevName[d.Severity]
		if sev == "" {
			sev = "unknown"
		}
		// 1-based for display
		fmt.Fprintf(&sb, "line %d: [%s] %s\n", d.Range.Start.Line+1, sev, d.Message)
	}
	return strings.TrimRight(sb.String(), "\n")
}
