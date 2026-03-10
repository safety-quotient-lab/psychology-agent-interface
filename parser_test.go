package main

import (
	"strings"
	"testing"
)

func TestParseNativeSingle(t *testing.T) {
	text := `<tool_call>{"name":"shell","arguments":{"cmd":"ls -la"}}</tool_call>`
	calls := parseNative(text)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Name != "shell" {
		t.Errorf("expected name 'shell', got %q", calls[0].Name)
	}
	if calls[0].Args["cmd"] != "ls -la" {
		t.Errorf("expected cmd 'ls -la', got %v", calls[0].Args["cmd"])
	}
}

func TestParseNativeMulti(t *testing.T) {
	text := `<tool_call>{"name":"shell","arguments":{"cmd":"echo hi"}}</tool_call>
Some text between calls.
<tool_call>{"name":"read_file","arguments":{"path":"/tmp/foo.txt"}}</tool_call>`
	calls := parseNative(text)
	if len(calls) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(calls))
	}
	if calls[1].Name != "read_file" {
		t.Errorf("expected 'read_file', got %q", calls[1].Name)
	}
	if calls[1].Args["path"] != "/tmp/foo.txt" {
		t.Errorf("expected path '/tmp/foo.txt', got %v", calls[1].Args["path"])
	}
}

func TestParseNativeEmpty(t *testing.T) {
	calls := parseNative("No tool calls here, just a plain response.")
	if len(calls) != 0 {
		t.Fatalf("expected 0 calls, got %d", len(calls))
	}
}

func TestParseNativeNoArgs(t *testing.T) {
	// arguments field omitted — should still parse with empty map
	text := `<tool_call>{"name":"shell"}</tool_call>`
	calls := parseNative(text)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Args == nil {
		t.Error("expected non-nil Args map")
	}
}

func TestParseReactSingle(t *testing.T) {
	text := `I need to list the files.
TOOL_CALL: {"name":"list_files","arguments":{"pattern":"**/*.go"}}`
	calls := parseReact(text)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Name != "list_files" {
		t.Errorf("expected 'list_files', got %q", calls[0].Name)
	}
	if calls[0].Args["pattern"] != "**/*.go" {
		t.Errorf("expected pattern '**/*.go', got %v", calls[0].Args["pattern"])
	}
}

func TestParseReactNestedJSON(t *testing.T) {
	// arguments value contains nested braces — must decode correctly
	text := `TOOL_CALL: {"name":"write_file","arguments":{"path":"x.json","content":"{\"key\":\"val\"}"}}`
	calls := parseReact(text)
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Args["path"] != "x.json" {
		t.Errorf("expected path 'x.json', got %v", calls[0].Args["path"])
	}
}

func TestParseReactEmpty(t *testing.T) {
	calls := parseReact("Here is my final answer with no tool calls.")
	if len(calls) != 0 {
		t.Fatalf("expected 0 calls, got %d", len(calls))
	}
}

func TestStripMarkupNative(t *testing.T) {
	text := `<tool_call>{"name":"shell","arguments":{"cmd":"ls"}}</tool_call>Here is the result.`
	result := stripMarkup(text)
	if result != "Here is the result." {
		t.Errorf("unexpected result: %q", result)
	}
}

func TestStripMarkupReact(t *testing.T) {
	// Strip removes the TOOL_CALL: marker; preserve surrounding text.
	// Note: non-greedy \{.*?\} stops at first '}', which matches Python's behavior.
	text := "TOOL_CALL: {\"name\":\"shell\",\"arguments\":{}}\nAnd the output was: foo"
	result := stripMarkup(text)
	if strings.Contains(result, "TOOL_CALL:") {
		t.Errorf("TOOL_CALL: should be stripped, got %q", result)
	}
	if !strings.Contains(result, "And the output was: foo") {
		t.Errorf("expected surrounding text preserved, got %q", result)
	}
}

func TestStripMarkupClean(t *testing.T) {
	text := "  No markup here.  "
	result := stripMarkup(text)
	if result != "No markup here." {
		t.Errorf("unexpected result: %q", result)
	}
}
