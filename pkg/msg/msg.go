// Package msg defines the shared Message type used across all pai packages.
// Eliminates the need for duplicate Message definitions in prompt, session,
// and main packages.
package msg

import (
	"regexp"
	"strings"
)

// Message represents a conversation message (system, user, assistant, or tool).
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

var nativeStripRe = regexp.MustCompile(`(?s)<tool_call>.*?</tool_call>`)

// stripReactToolCall removes a single TOOL_CALL: {...} block starting at pos,
// using brace-depth counting to handle nested JSON correctly.
func stripReactToolCall(text string, pos int) (string, bool) {
	braceStart := strings.IndexByte(text[pos:], '{')
	if braceStart < 0 {
		return text, false
	}
	braceStart += pos

	depth := 0
	inString := false
	escape := false
	for i := braceStart; i < len(text); i++ {
		ch := text[i]
		if escape {
			escape = false
			continue
		}
		if ch == '\\' && inString {
			escape = true
			continue
		}
		if ch == '"' {
			inString = !inString
			continue
		}
		if inString {
			continue
		}
		if ch == '{' {
			depth++
		} else if ch == '}' {
			depth--
			if depth == 0 {
				return text[:pos] + text[i+1:], true
			}
		}
	}
	// Unclosed brace — strip from TOOL_CALL: to end
	return text[:pos], true
}

// StripMarkup removes tool call markup from assistant text.
func StripMarkup(text string) string {
	// Native: <tool_call>...</tool_call> (regex handles nested content fine
	// because the closing tag is unambiguous markup, not JSON)
	text = nativeStripRe.ReplaceAllString(text, "")

	// ReAct: TOOL_CALL: {...} — use depth-aware stripping for nested JSON
	for {
		pos := strings.Index(text, "TOOL_CALL:")
		if pos < 0 {
			break
		}
		var found bool
		text, found = stripReactToolCall(text, pos)
		if !found {
			break
		}
	}

	return strings.TrimSpace(text)
}
