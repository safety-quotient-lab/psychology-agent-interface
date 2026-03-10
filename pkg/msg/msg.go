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

var (
	nativeStripRe = regexp.MustCompile(`(?s)<tool_call>.*?</tool_call>`)
	reactStripRe  = regexp.MustCompile(`(?s)TOOL_CALL:\s*\{.*?\}`)
)

// StripMarkup removes tool call markup from assistant text.
func StripMarkup(text string) string {
	text = nativeStripRe.ReplaceAllString(text, "")
	text = reactStripRe.ReplaceAllString(text, "")
	return strings.TrimSpace(text)
}
