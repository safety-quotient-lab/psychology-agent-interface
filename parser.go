package main

import (
	"encoding/json"
	"regexp"
	"strings"

	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/msg"
)

// ToolCall represents a parsed tool invocation.
type ToolCall struct {
	Name string
	Args map[string]any
}

var nativeRe = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)

// parseNative parses <tool_call>{...}</tool_call> blocks (Qwen2.5 native format).
func parseNative(text string) []ToolCall {
	var calls []ToolCall
	for _, m := range nativeRe.FindAllStringSubmatch(text, -1) {
		var obj struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(m[1]), &obj); err != nil || obj.Name == "" {
			continue
		}
		if obj.Arguments == nil {
			obj.Arguments = map[string]any{}
		}
		calls = append(calls, ToolCall{Name: obj.Name, Args: obj.Arguments})
	}
	return calls
}

var reactRe = regexp.MustCompile(`TOOL_CALL:\s*(\{)`)

// parseReact parses TOOL_CALL: {...} blocks (ReAct fallback format).
// Uses json.Decoder.Decode to correctly handle nested JSON objects.
func parseReact(text string) []ToolCall {
	var calls []ToolCall
	for _, loc := range reactRe.FindAllStringSubmatchIndex(text, -1) {
		// loc[2]/loc[3] is the capture group for '{' position
		start := loc[2]
		dec := json.NewDecoder(strings.NewReader(text[start:]))
		var obj struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if err := dec.Decode(&obj); err != nil || obj.Name == "" {
			continue
		}
		if obj.Arguments == nil {
			obj.Arguments = map[string]any{}
		}
		calls = append(calls, ToolCall{Name: obj.Name, Args: obj.Arguments})
	}
	return calls
}

// stripMarkup removes tool call markup from assistant text.
var stripMarkup = msg.StripMarkup
