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

// contentBlock mirrors claude-replay's expected content block structure.
type contentBlock struct {
	Type  string `json:"type"`
	Text  string `json:"text,omitempty"`
	Name  string `json:"name,omitempty"`
	Input any    `json:"input,omitempty"`
}

// toolResultBlock represents a tool result content block.
type toolResultBlock struct {
	Type      string `json:"type"`
	ToolUseID string `json:"tool_use_id"`
	Content   string `json:"content"`
}

// isPriming returns true if the message looks like a few-shot priming example.
// Priming messages use specific patterns injected by PsychologyFewShot.
func isPriming(msg Message) bool {
	if msg.Role == "user" && strings.HasPrefix(msg.Content, "What ") && len(msg.Content) < 100 {
		return true
	}
	if msg.Role == "assistant" && strings.Contains(msg.Content, "[observation]") && strings.Contains(msg.Content, "Confidence:") {
		// Check if it looks like the canned example (short, formatted)
		if len(msg.Content) < 500 {
			return true
		}
	}
	return false
}

// isToolCallOnly returns true if the content contains only tool call markup
// with no meaningful surrounding text.
func isToolCallOnly(content string) bool {
	clean := pmsg.StripMarkup(content)
	return strings.TrimSpace(clean) == ""
}

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
	toolUseCounter := 0
	msgTime := time.Now().Add(-time.Duration(len(conv)) * 30 * time.Second)

	// Skip priming messages at the start (after system prompt)
	startIdx := 0
	for i, msg := range conv {
		if msg.Role == "system" {
			startIdx = i + 1
			continue
		}
		if isPriming(msg) {
			startIdx = i + 1
			continue
		}
		break
	}

	for i := startIdx; i < len(conv); i++ {
		msg := conv[i]
		if msg.Role == "system" {
			continue
		}

		uuid := fmt.Sprintf("%s-%04d", ts, i)
		parentUUID := prevUUID
		if parentUUID == "" {
			parentUUID = "null-00000000-0000-0000-0000-000000000000"
		}

		var roleType string
		var content any
		var msgRole string

		switch msg.Role {
		case "user":
			if msg.Name != "" {
				// Tool result returned as user message (ReAct format)
				roleType = "tool_result"
				msgRole = "tool"
				toolUseCounter++
				content = []toolResultBlock{{
					Type:      "tool_result",
					ToolUseID: fmt.Sprintf("tool_%s_%04d", ts, toolUseCounter),
					Content:   msg.Content,
				}}
			} else {
				roleType = "user"
				msgRole = "user"
				content = msg.Content
			}

		case "tool":
			// Tool result message
			roleType = "tool_result"
			msgRole = "tool"
			toolUseCounter++
			content = []toolResultBlock{{
				Type:      "tool_result",
				ToolUseID: fmt.Sprintf("tool_%s_%04d", ts, toolUseCounter),
				Content:   msg.Content,
			}}

		case "assistant":
			roleType = "assistant"
			msgRole = "assistant"
			clean := pmsg.StripMarkup(msg.Content)
			if isToolCallOnly(msg.Content) {
				// Assistant message contains only a tool call — represent as
				// a tool_use content block so claude-replay renders it properly.
				toolUseCounter++
				content = []contentBlock{{
					Type: "tool_use",
					Name: extractToolName(msg.Content),
					Input: map[string]string{
						"raw": msg.Content,
					},
				}}
			} else if clean != "" {
				content = []contentBlock{{Type: "text", Text: clean}}
			} else {
				content = []contentBlock{{Type: "text", Text: msg.Content}}
			}

		default:
			continue
		}

		line := map[string]any{
			"parentUuid":  parentUUID,
			"isSidechain": false,
			"userType":    "external",
			"cwd":         cwd,
			"sessionId":   sessionID,
			"version":     "pai",
			"gitBranch":   "main",
			"type":        roleType,
			"message": map[string]any{
				"role":    msgRole,
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

// extractToolName pulls the tool name from raw tool-call markup.
func extractToolName(content string) string {
	// Native: <tool_call>{"name":"X",...}</tool_call>
	if idx := strings.Index(content, `"name"`); idx >= 0 {
		rest := content[idx+7:] // skip `"name":`
		rest = strings.TrimLeft(rest, " ")
		if len(rest) > 0 && rest[0] == '"' {
			end := strings.IndexByte(rest[1:], '"')
			if end >= 0 {
				return rest[1 : end+1]
			}
		}
	}
	return "unknown"
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
			continue // skip system prompt from markdown export
		case "user":
			if msg.Name == "" {
				sb.WriteString(fmt.Sprintf("**You:** %s\n\n", msg.Content))
			} else {
				sb.WriteString(fmt.Sprintf("*[tool result — %s]*\n```\n%s\n```\n\n", msg.Name, msg.Content))
			}
		case "assistant":
			clean := pmsg.StripMarkup(msg.Content)
			if clean == "" {
				clean = msg.Content
			}
			sb.WriteString(fmt.Sprintf("**Assistant:** %s\n\n", clean))
		case "tool":
			sb.WriteString(fmt.Sprintf("*[tool result — %s]*\n```\n%s\n```\n\n", msg.Name, msg.Content))
		}
	}

	return path, os.WriteFile(path, []byte(sb.String()), 0644)
}
