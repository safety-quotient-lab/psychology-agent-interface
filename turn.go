package main

import "fmt"

// Turn consolidates the parse → execute → format pipeline for tool calls.
// Both TUI (model.go) and headless (print.go) use the same Turn to avoid
// duplicating native/ReAct branching logic (Plan 9 Phase 3: message plumbing).
type Turn struct {
	Native bool   // true = native <tool_call> format; false = ReAct TOOL_CALL:
	CWD    string // working directory for tool execution
}

// ParseCalls extracts tool calls from an assistant reply.
// Returns nil when the reply contains no tool invocations (final answer).
func (t Turn) ParseCalls(reply string) []ToolCall {
	if t.Native {
		return parseNative(reply)
	}
	return parseReact(reply)
}

// FormatResult wraps a tool result into the correct message shape.
func (t Turn) FormatResult(name, result string) Message {
	if t.Native {
		return Message{Role: "tool", Name: name, Content: result}
	}
	return Message{
		Role:    "user",
		Content: fmt.Sprintf("TOOL_RESULT (%s):\n%s\n\nContinue.", name, result),
	}
}

// FormatDenial wraps a tool denial into the correct message shape.
func (t Turn) FormatDenial(name string) Message {
	denial := fmt.Sprintf("Tool call '%s' was denied by the user.", name)
	return t.FormatResult(name, denial)
}

// ExecuteAll runs all tool calls and returns result messages.
// Used in headless mode where no approval step intervenes.
func (t Turn) ExecuteAll(calls []ToolCall) []Message {
	msgs := make([]Message, 0, len(calls))
	for _, call := range calls {
		result := executeTool(call.Name, call.Args, t.CWD)
		msgs = append(msgs, t.FormatResult(call.Name, result))
	}
	return msgs
}

// ProcessReply parses a reply and executes any tool calls found.
// Returns (toolResultMessages, true) if tools ran, or (nil, false) for
// a final answer with no tool calls.
func (t Turn) ProcessReply(reply string) ([]Message, bool) {
	calls := t.ParseCalls(reply)
	if len(calls) == 0 {
		return nil, false
	}
	return t.ExecuteAll(calls), true
}
