package main

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/bmatcuk/doublestar/v4"
)

var (
	htmlScriptRe  = regexp.MustCompile(`(?si)<(script|style)[^>]*>.*?</(script|style)>`)
	htmlTagRe     = regexp.MustCompile(`<[^>]+>`)
	htmlSpaceRe   = regexp.MustCompile(`[ \t]+`)
	htmlNewlineRe = regexp.MustCompile(`\n{3,}`)
)

func stripHTML(s string) string {
	s = htmlScriptRe.ReplaceAllString(s, "")
	s = htmlTagRe.ReplaceAllString(s, " ")
	s = html.UnescapeString(s)
	s = htmlSpaceRe.ReplaceAllString(s, " ")
	s = strings.ReplaceAll(s, "\r", "\n")
	s = htmlNewlineRe.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}

const truncLimit = 4000

// isShellCommand returns true if the first word of text is a binary on PATH
// or an explicit path reference (./foo, /usr/bin/foo).
func isShellCommand(text string) bool {
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return false
	}
	word := fields[0]
	if strings.HasPrefix(word, "./") || strings.HasPrefix(word, "/") {
		return true
	}
	_, err := exec.LookPath(word)
	return err == nil
}

// runDirectShell runs text as a bash command in cwd and returns combined output.
func runDirectShell(text, cwd string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	c := exec.CommandContext(ctx, "bash", "-c", text)
	c.Dir = cwd
	out, err := c.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return "ERROR: timed out"
	}
	result := string(out)
	if err != nil && result == "" {
		result = err.Error()
	}
	if result == "" {
		return "(no output)"
	}
	return truncate(result)
}

// wordWrap wraps plain text (no ANSI codes) at width columns.
func wordWrap(s string, width int) string {
	if width <= 0 {
		return s
	}
	var out strings.Builder
	for i, line := range strings.Split(s, "\n") {
		if i > 0 {
			out.WriteByte('\n')
		}
		if len(line) <= width {
			out.WriteString(line)
			continue
		}
		col := 0
		for j, word := range strings.Fields(line) {
			wl := len(word)
			if j == 0 {
				out.WriteString(word)
				col = wl
			} else if col+1+wl > width {
				out.WriteByte('\n')
				out.WriteString(word)
				col = wl
			} else {
				out.WriteByte(' ')
				out.WriteString(word)
				col += 1 + wl
			}
		}
	}
	return out.String()
}

func truncate(s string) string {
	if len(s) > truncLimit {
		return s[:truncLimit] + "\n[truncated]"
	}
	return s
}

// executeTool runs the named tool with the given args relative to cwd.
func executeTool(name string, args map[string]any, cwd string) string {
	return executeToolInner(name, args, cwd)
}

func executeToolInner(name string, args map[string]any, cwd string) string {
	switch name {

	case "shell":
		cmd, _ := args["cmd"].(string)
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		c := exec.CommandContext(ctx, "bash", "-c", cmd)
		c.Dir = cwd
		out, err := c.CombinedOutput()
		if ctx.Err() == context.DeadlineExceeded {
			return "ERROR: timed out"
		}
		result := string(out)
		if err != nil && result == "" {
			result = err.Error()
		}
		if result == "" {
			return "(no output)"
		}
		return truncate(result)

	case "read_file":
		path, _ := args["path"].(string)
		if !filepath.IsAbs(path) {
			path = filepath.Join(cwd, path)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Sprintf("ERROR: %v", err)
		}
		return truncate(string(data))

	case "write_file":
		path, _ := args["path"].(string)
		content, _ := args["content"].(string)
		if !filepath.IsAbs(path) {
			path = filepath.Join(cwd, path)
		}
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return fmt.Sprintf("ERROR: %v", err)
		}
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return fmt.Sprintf("ERROR: %v", err)
		}
		return fmt.Sprintf("Wrote %d bytes to %s", len(content), path)

	case "edit_file":
		path, _ := args["path"].(string)
		oldStr, _ := args["old_str"].(string)
		newStr, _ := args["new_str"].(string)
		if !filepath.IsAbs(path) {
			path = filepath.Join(cwd, path)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			if os.IsNotExist(err) {
				return fmt.Sprintf("ERROR: file does not exist: %s — use write_file to create it first", path)
			}
			return fmt.Sprintf("ERROR: %v", err)
		}
		content := string(data)
		if !strings.Contains(content, oldStr) {
			return "ERROR: old_str not found in file (check whitespace and exact match)"
		}
		updated := strings.Replace(content, oldStr, newStr, 1)
		if err := os.WriteFile(path, []byte(updated), 0644); err != nil {
			return fmt.Sprintf("ERROR: %v", err)
		}
		return fmt.Sprintf("Edited %s (%d bytes → %d bytes)", path, len(content), len(updated))

	case "list_files":
		pattern, _ := args["pattern"].(string)
		fsys := os.DirFS(cwd)
		matches, err := doublestar.Glob(fsys, pattern)
		if err != nil {
			return fmt.Sprintf("ERROR: %v", err)
		}
		if len(matches) == 0 {
			return "(no matches)"
		}
		return strings.Join(matches, "\n")

	case "fetch_url":
		rawURL, _ := args["url"].(string)
		client := &http.Client{Timeout: 15 * time.Second}
		resp, err := client.Get(rawURL)
		if err != nil {
			return fmt.Sprintf("ERROR: %v", err)
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Sprintf("ERROR: reading response: %v", err)
		}
		ct := resp.Header.Get("Content-Type")
		if strings.Contains(ct, "text/html") || ct == "" {
			return truncate(stripHTML(string(body)))
		}
		return truncate(string(body))

	case "web_search":
		query, _ := args["query"].(string)
		apiKey := os.Getenv("KAGI_API_KEY")
		if apiKey == "" {
			return "ERROR: KAGI_API_KEY not set (add to ~/.dev.vars)"
		}
		limit := 5
		if l, ok := args["limit"].(float64); ok && l > 0 {
			limit = int(l)
		}
		reqURL := fmt.Sprintf("https://kagi.com/api/v0/search?q=%s&limit=%d",
			url.QueryEscape(query), limit)
		req, err := http.NewRequestWithContext(context.Background(), "GET", reqURL, nil)
		if err != nil {
			return fmt.Sprintf("ERROR: %v", err)
		}
		req.Header.Set("Authorization", "Bot "+apiKey)
		client := &http.Client{Timeout: 15 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			return fmt.Sprintf("ERROR: %v", err)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		var kagiResp struct {
			Data []struct {
				T       int    `json:"t"`
				Rank    int    `json:"rank"`
				URL     string `json:"url"`
				Title   string `json:"title"`
				Snippet string `json:"snippet"`
			} `json:"data"`
			Error []struct {
				Msg string `json:"msg"`
			} `json:"error"`
		}
		if err := json.Unmarshal(body, &kagiResp); err != nil {
			return fmt.Sprintf("ERROR: parse response: %v", err)
		}
		if len(kagiResp.Error) > 0 {
			return fmt.Sprintf("ERROR: %s", kagiResp.Error[0].Msg)
		}
		var sb strings.Builder
		for _, item := range kagiResp.Data {
			if item.T != 0 {
				continue // skip non-result entries (ads, related searches, etc.)
			}
			fmt.Fprintf(&sb, "[%d] %s\n%s\n%s\n\n", item.Rank, item.Title, item.URL, item.Snippet)
		}
		if sb.Len() == 0 {
			return "(no results)"
		}
		return truncate(sb.String())

	case "search":
		pattern, _ := args["pattern"].(string)
		path, _ := args["path"].(string)
		if path == "" {
			path = "."
		}
		if !filepath.IsAbs(path) {
			path = filepath.Join(cwd, path)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		var out []byte
		if _, err := exec.LookPath("rg"); err == nil {
			out, _ = exec.CommandContext(ctx, "rg", "-n", pattern, path).Output()
		} else {
			out, _ = exec.CommandContext(ctx, "grep", "-rn", pattern, path).Output()
		}
		if ctx.Err() == context.DeadlineExceeded {
			return "ERROR: timed out"
		}
		result := strings.TrimRight(string(out), "\n")
		if result == "" {
			return "(no matches)"
		}
		return truncate(result)

	default:
		return fmt.Sprintf("ERROR: unknown tool '%s'", name)
	}
}
