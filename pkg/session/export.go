package session

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	pmsg "github.com/safety-quotient-lab/psychology-agent-interface/pkg/msg"
)

// ExportJSONL writes the conversation as a Claude Code-compatible JSONL file
// so claude-replay can generate an HTML replay from it.
func ExportJSONL(model string, conv []Message, cwd string) (string, error) {
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
				content = fmt.Sprintf("[tool result: %s]\n%s", msg.Name, msg.Content)
			} else {
				content = msg.Content
			}
		case "tool":
			roleType = "user"
			content = fmt.Sprintf("[tool result: %s]\n%s", msg.Name, msg.Content)
		case "assistant":
			roleType = "assistant"
			clean := pmsg.StripMarkup(msg.Content)
			if clean == "" {
				clean = msg.Content
			}
			content = []contentBlock{{Type: "text", Text: clean}}
		default:
			continue
		}

		line := map[string]interface{}{
			"parentUuid":  parentUUID,
			"isSidechain": false,
			"userType":    "external",
			"cwd":         cwd,
			"sessionId":   sessionID,
			"version":     "pai",
			"gitBranch":   "main",
			"type":        roleType,
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

// GenerateReplay runs claude-replay on jsonlPath and writes HTML to htmlPath.
func GenerateReplay(jsonlPath, htmlPath, title string) error {
	cmd := exec.Command("claude-replay", jsonlPath,
		"--theme", "tokyo-night",
		"--title", title,
		"-o", htmlPath,
	)
	return cmd.Run()
}

// ExportMarkdown writes the conversation as a Markdown file to cwd.
func ExportMarkdown(model string, conv []Message, cwd string) (string, error) {
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
			clean := pmsg.StripMarkup(msg.Content)
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
