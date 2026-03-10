package tool

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

const truncLimit = 4000

func truncate(s string) string {
	if len(s) > truncLimit {
		return s[:truncLimit] + "\n[truncated]"
	}
	return s
}

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

// RegisterBuiltins adds all standard tools to the registry.
// Pass kagiKey="" to exclude web_search.
func RegisterBuiltins(r *Registry, kagiKey string) {
	r.Register(Tool{
		Name:        "shell",
		Description: "Execute a bash command. Returns combined stdout and stderr.",
		Parameters:  []Param{{Name: "cmd", Type: "string", Description: "Bash command to run", Required: true}},
		Execute: func(args map[string]any, cwd string) string {
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
		},
	})

	r.Register(Tool{
		Name:        "read_file",
		Description: "Read the full contents of a file.",
		Parameters:  []Param{{Name: "path", Type: "string", Description: "Absolute or relative file path", Required: true}},
		Execute: func(args map[string]any, cwd string) string {
			path, _ := args["path"].(string)
			if !filepath.IsAbs(path) {
				path = filepath.Join(cwd, path)
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return fmt.Sprintf("ERROR: %v", err)
			}
			return truncate(string(data))
		},
	})

	r.Register(Tool{
		Name:        "write_file",
		Description: "Write content to a file (overwrites if it exists). Use edit_file instead for targeted changes.",
		Parameters: []Param{
			{Name: "path", Type: "string", Required: true},
			{Name: "content", Type: "string", Required: true},
		},
		Execute: func(args map[string]any, cwd string) string {
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
		},
	})

	r.Register(Tool{
		Name:        "edit_file",
		Description: "Replace the first occurrence of old_str with new_str in a file. old_str must match exactly (whitespace matters). Prefer this over write_file for targeted edits.",
		Parameters: []Param{
			{Name: "path", Type: "string", Description: "Absolute or relative file path", Required: true},
			{Name: "old_str", Type: "string", Description: "Exact string to find and replace", Required: true},
			{Name: "new_str", Type: "string", Description: "Replacement string", Required: true},
		},
		Execute: func(args map[string]any, cwd string) string {
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
		},
	})

	r.Register(Tool{
		Name:        "list_files",
		Description: "List files matching a glob pattern. Use '*.ext' for current directory only. Only use '**/*.ext' when you explicitly need to recurse into subdirectories.",
		Parameters:  []Param{{Name: "pattern", Type: "string", Description: "Glob pattern, e.g. '*.py' for current dir or '**/*.py' for recursive", Required: true}},
		Execute: func(args map[string]any, cwd string) string {
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
		},
	})

	r.Register(Tool{
		Name:        "search",
		Description: "Search for a regex pattern in files (like grep -rn).",
		Parameters: []Param{
			{Name: "pattern", Type: "string", Description: "Regex to search for", Required: true},
			{Name: "path", Type: "string", Description: "File or directory to search", Default: "."},
		},
		Execute: func(args map[string]any, cwd string) string {
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
		},
	})

	r.Register(Tool{
		Name:        "fetch_url",
		Description: "Fetch a URL and return its content as plain text. Use for documentation, GitHub issues, or any web link the user provides.",
		Parameters:  []Param{{Name: "url", Type: "string", Description: "The URL to fetch", Required: true}},
		Execute: func(args map[string]any, cwd string) string {
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
		},
	})

	if kagiKey != "" {
		r.Register(Tool{
			Name:        "web_search",
			Description: "Search the web via Kagi. Returns ranked results with title, URL, and snippet.",
			Parameters: []Param{
				{Name: "query", Type: "string", Description: "Search query", Required: true},
				{Name: "limit", Type: "integer", Description: "Max results (default 5)"},
			},
			Execute: func(args map[string]any, cwd string) string {
				query, _ := args["query"].(string)
				apiKey := os.Getenv("KAGI_API_KEY")
				if apiKey == "" {
					return "ERROR: web_search unavailable — KAGI_API_KEY not set. Do NOT retry this tool. Answer without web search or tell the user to set KAGI_API_KEY in ~/.dev.vars."
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
						continue
					}
					fmt.Fprintf(&sb, "[%d] %s\n%s\n%s\n\n", item.Rank, item.Title, item.URL, item.Snippet)
				}
				if sb.Len() == 0 {
					return "(no results)"
				}
				return truncate(sb.String())
			},
		})
	}

	r.Register(Tool{
		Name:        "ask_user",
		Description: "Ask the user a clarifying question and wait for their answer. Use when you need information you cannot determine from context.",
		Parameters:  []Param{{Name: "question", Type: "string", Description: "The question to ask the user", Required: true}},
		Execute: func(args map[string]any, cwd string) string {
			// ask_user handled specially by the TUI — returns the question as-is.
			question, _ := args["question"].(string)
			return question
		},
	})
}

// IsShellCommand returns true if the first word of text appears on PATH
// or starts with ./ or /.
func IsShellCommand(text string) bool {
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

// RunDirectShell runs text as a bash command in cwd and returns combined output.
func RunDirectShell(text, cwd string) string {
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
