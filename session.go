package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Session is a saved conversation snapshot.
type Session struct {
	ID           string    `json:"id"`
	Model        string    `json:"model"`
	SavedAt      time.Time `json:"saved_at"`
	Conversation []Message `json:"conversation"`
}

func sessionDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".local", "share", "pai", "sessions")
	return dir, os.MkdirAll(dir, 0755)
}

func saveSession(model string, conv []Message) (string, error) {
	dir, err := sessionDir()
	if err != nil {
		return "", err
	}
	id := time.Now().Format("20060102-150405")
	s := Session{ID: id, Model: model, SavedAt: time.Now(), Conversation: conv}
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, id+".json")
	return path, os.WriteFile(path, b, 0644)
}

func listSessions() ([]Session, error) {
	dir, err := sessionDir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var sessions []Session
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		var s Session
		if json.Unmarshal(data, &s) == nil {
			sessions = append(sessions, s)
		}
	}
	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].SavedAt.After(sessions[j].SavedAt)
	})
	return sessions, nil
}

func loadSessionByIndex(n int) (Session, error) {
	sessions, err := listSessions()
	if err != nil {
		return Session{}, err
	}
	if n < 1 || n > len(sessions) {
		return Session{}, os.ErrNotExist
	}
	return sessions[n-1], nil
}

// exportJSONL writes the conversation as a Claude Code-compatible JSONL file
// so claude-replay can generate an HTML replay from it.
func exportJSONL(model string, conv []Message, cwd string) (string, error) {
	ts := time.Now().Format("20060102-150405")
	fname := "session-" + ts + ".jsonl"
	path := filepath.Join(cwd, fname)

	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	sessionID := ts

	// Build a fake project-slug from cwd (mirrors Claude Code convention)
	slug := strings.ReplaceAll(cwd, "/", "-")

	var prevUUID string
	msgTime := time.Now().Add(-time.Duration(len(conv)) * 30 * time.Second)

	for i, msg := range conv {
		if msg.Role == "system" {
			continue
		}

		uuid := fmt.Sprintf("%s-%04d", ts, i)
		parentUUID := prevUUID
		if parentUUID == "" {
			parentUUID = "null-00000000-0000-0000-0000-000000000000"
		}

		type contentBlock struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}

		var roleType string
		var content interface{}

		switch msg.Role {
		case "user":
			roleType = "user"
			if msg.Name != "" {
				// Tool result — show as tool output block
				content = fmt.Sprintf("[tool result: %s]\n%s", msg.Name, msg.Content)
			} else {
				content = msg.Content
			}
		case "tool":
			roleType = "user"
			content = fmt.Sprintf("[tool result: %s]\n%s", msg.Name, msg.Content)
		case "assistant":
			roleType = "assistant"
			clean := stripMarkup(msg.Content)
			if clean == "" {
				clean = msg.Content
			}
			content = []contentBlock{{Type: "text", Text: clean}}
		default:
			continue
		}

		line := map[string]interface{}{
			"parentUuid": parentUUID,
			"isSidechain": false,
			"userType":   "external",
			"cwd":        cwd,
			"sessionId":  sessionID,
			"version":    "pai",
			"gitBranch":  "main",
			"type":       roleType,
			"message": map[string]interface{}{
				"role":    msg.Role,
				"content": content,
				"model":   model,
			},
			"uuid":      uuid,
			"timestamp": msgTime.UTC().Format(time.RFC3339Nano),
			"_project":  slug,
		}
		if err := enc.Encode(line); err != nil {
			return "", err
		}
		prevUUID = uuid
		msgTime = msgTime.Add(30 * time.Second)
	}

	return path, nil
}

// generateReplay runs claude-replay on jsonlPath and writes HTML to htmlPath.
// Returns the HTML path on success.
func generateReplay(jsonlPath, htmlPath, title string) error {
	cmd := exec.Command("claude-replay", jsonlPath,
		"--theme", "tokyo-night",
		"--title", title,
		"-o", htmlPath,
	)
	return cmd.Run()
}

// exportMarkdown writes the conversation as a Markdown file to cwd.
// Returns the path of the written file.
func exportMarkdown(model string, conv []Message, cwd string) (string, error) {
	ts := time.Now().Format("20060102-150405")
	fname := "session-" + ts + ".md"
	path := filepath.Join(cwd, fname)

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# pai session — %s — %s\n\n",
		model, time.Now().Format("2006-01-02 15:04:05")))

	for _, msg := range conv {
		switch msg.Role {
		case "system":
			sb.WriteString(fmt.Sprintf("*[system: %s]*\n\n", msg.Content))
		case "user":
			if msg.Name == "" {
				sb.WriteString(fmt.Sprintf("**You:** %s\n\n", msg.Content))
			} else {
				sb.WriteString(fmt.Sprintf("*[tool result — %s]*\n\n", msg.Name))
			}
		case "assistant":
			clean := stripMarkup(msg.Content)
			if clean == "" {
				clean = msg.Content
			}
			sb.WriteString(fmt.Sprintf("**Assistant:** %s\n\n", clean))
		case "tool":
			sb.WriteString(fmt.Sprintf("*[tool result — %s: %s]*\n\n", msg.Name, msg.Content))
		}
	}

	return path, os.WriteFile(path, []byte(sb.String()), 0644)
}
