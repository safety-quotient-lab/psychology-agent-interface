package main

import (
	"strings"
	"testing"
)

func TestTurnFormatResultNative(t *testing.T) {
	tr := Turn{Native: true, CWD: "/tmp"}
	msg := tr.FormatResult("shell", "hello")
	if msg.Role != "tool" {
		t.Errorf("expected role 'tool', got %q", msg.Role)
	}
	if msg.Name != "shell" {
		t.Errorf("expected name 'shell', got %q", msg.Name)
	}
	if msg.Content != "hello" {
		t.Errorf("expected content 'hello', got %q", msg.Content)
	}
}

func TestTurnFormatResultReact(t *testing.T) {
	tr := Turn{Native: false, CWD: "/tmp"}
	msg := tr.FormatResult("shell", "hello")
	if msg.Role != "user" {
		t.Errorf("expected role 'user', got %q", msg.Role)
	}
	if !strings.Contains(msg.Content, "TOOL_RESULT (shell)") {
		t.Errorf("expected TOOL_RESULT header, got %q", msg.Content)
	}
	if !strings.Contains(msg.Content, "hello") {
		t.Errorf("expected result in content, got %q", msg.Content)
	}
}

func TestTurnFormatDenial(t *testing.T) {
	tr := Turn{Native: true, CWD: "/tmp"}
	msg := tr.FormatDenial("shell")
	if !strings.Contains(msg.Content, "denied") {
		t.Errorf("expected denial message, got %q", msg.Content)
	}
}

func TestTurnProcessReplyNoTools(t *testing.T) {
	tr := Turn{Native: false, CWD: "/tmp"}
	results, hadTools := tr.ProcessReply("Just a plain answer.")
	if hadTools {
		t.Error("expected no tools for plain reply")
	}
	if results != nil {
		t.Error("expected nil results")
	}
}

func TestTurnProcessReplyWithReactTool(t *testing.T) {
	tr := Turn{Native: false, CWD: "/tmp"}
	reply := `Let me check.
TOOL_CALL: {"name": "shell", "arguments": {"cmd": "echo hello"}}`
	results, hadTools := tr.ProcessReply(reply)
	if !hadTools {
		t.Error("expected tools found")
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if !strings.Contains(results[0].Content, "hello") {
		t.Errorf("expected shell output, got %q", results[0].Content)
	}
}
