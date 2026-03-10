package session

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExportJSONLSkipsPriming(t *testing.T) {
	conv := []Message{
		{Role: "system", Content: "system prompt"},
		{Role: "user", Name: "_priming", Content: "What causes anxiety?"},
		{Role: "assistant", Name: "_priming", Content: "[observation] test\n\nConfidence: high"},
		{Role: "user", Content: "Hello, tell me about stress"},
		{Role: "assistant", Content: "Stress affects everyone differently."},
	}
	dir := t.TempDir()
	path, err := ExportJSONL("test-model", conv, dir)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(path)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	// Should export 2 messages: the real user message + assistant reply
	// (system, priming user, priming assistant all skipped)
	if len(lines) != 2 {
		t.Errorf("expected 2 lines, got %d", len(lines))
		for i, l := range lines {
			t.Logf("  line %d: %s", i, l[:min(len(l), 100)])
		}
	}
}

func TestExportJSONLToolCallAsToolUse(t *testing.T) {
	conv := []Message{
		{Role: "system", Content: "system prompt"},
		{Role: "user", Content: "list my files"},
		{Role: "assistant", Content: `TOOL_CALL: {"name":"list_files","arguments":{"pattern":"*"}}`},
		{Role: "user", Name: "list_files", Content: "a.go\nb.go"},
		{Role: "assistant", Content: "You have a.go and b.go."},
	}
	dir := t.TempDir()
	path, err := ExportJSONL("test-model", conv, dir)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(path)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 4 {
		t.Fatalf("expected 4 lines, got %d", len(lines))
	}

	// Line 1: assistant with tool_use block (not raw TOOL_CALL text)
	var assistantLine map[string]any
	json.Unmarshal([]byte(lines[1]), &assistantLine)
	msg := assistantLine["message"].(map[string]any)
	content := msg["content"].([]any)
	block := content[0].(map[string]any)
	if block["type"] != "tool_use" {
		t.Errorf("expected tool_use block, got %v", block["type"])
	}
	if block["name"] != "list_files" {
		t.Errorf("expected tool name list_files, got %v", block["name"])
	}

	// Line 2: tool result
	var toolLine map[string]any
	json.Unmarshal([]byte(lines[2]), &toolLine)
	if toolLine["type"] != "tool_result" {
		t.Errorf("expected type=tool_result, got %v", toolLine["type"])
	}

	// Line 3: assistant with clean text (no TOOL_CALL markup)
	var finalLine map[string]any
	json.Unmarshal([]byte(lines[3]), &finalLine)
	finalMsg := finalLine["message"].(map[string]any)
	finalContent := finalMsg["content"].([]any)
	finalBlock := finalContent[0].(map[string]any)
	if finalBlock["type"] != "text" {
		t.Errorf("expected text block, got %v", finalBlock["type"])
	}
	if finalBlock["text"] != "You have a.go and b.go." {
		t.Errorf("unexpected text: %v", finalBlock["text"])
	}
}

func TestExportJSONLNativeToolCall(t *testing.T) {
	conv := []Message{
		{Role: "system", Content: "system prompt"},
		{Role: "user", Content: "search for main"},
		{Role: "assistant", Content: `<tool_call>{"name":"search","arguments":{"pattern":"main"}}</tool_call>`},
		{Role: "tool", Name: "search", Content: "main.go:1:package main"},
		{Role: "assistant", Content: "Found main in main.go."},
	}
	dir := t.TempDir()
	path, err := ExportJSONL("test-model", conv, dir)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(path)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 4 {
		t.Fatalf("expected 4 lines, got %d", len(lines))
	}

	// Verify tool_use block extracted correctly
	var assistantLine map[string]any
	json.Unmarshal([]byte(lines[1]), &assistantLine)
	msg := assistantLine["message"].(map[string]any)
	content := msg["content"].([]any)
	block := content[0].(map[string]any)
	if block["type"] != "tool_use" {
		t.Errorf("expected tool_use, got %v", block["type"])
	}
	if block["name"] != "search" {
		t.Errorf("expected search, got %v", block["name"])
	}
}

func TestExtractToolName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{`TOOL_CALL: {"name":"shell","arguments":{"cmd":"ls"}}`, "shell"},
		{`<tool_call>{"name":"read_file","arguments":{"path":"x"}}</tool_call>`, "read_file"},
		{`no tool here`, "unknown"},
	}
	for _, tt := range tests {
		got := extractToolName(tt.input)
		if got != tt.want {
			t.Errorf("extractToolName(%q) = %q, want %q", tt.input[:30], got, tt.want)
		}
	}
}

func TestExportJSONLOutputPath(t *testing.T) {
	conv := []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}
	dir := t.TempDir()
	path, err := ExportJSONL("test", conv, dir)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(filepath.Base(path), "session-") {
		t.Errorf("expected session- prefix, got %s", filepath.Base(path))
	}
	if !strings.HasSuffix(path, ".jsonl") {
		t.Errorf("expected .jsonl suffix, got %s", path)
	}
}
