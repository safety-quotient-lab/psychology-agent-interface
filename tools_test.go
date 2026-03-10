package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestShellEcho(t *testing.T) {
	result := executeTool("shell", map[string]any{"cmd": "echo hello"}, "/tmp")
	if strings.TrimSpace(result) != "hello" {
		t.Errorf("expected 'hello', got %q", result)
	}
}

func TestShellNoOutput(t *testing.T) {
	result := executeTool("shell", map[string]any{"cmd": "true"}, "/tmp")
	if result != "(no output)" {
		t.Errorf("expected '(no output)', got %q", result)
	}
}

func TestShellStderr(t *testing.T) {
	// CombinedOutput — stderr should appear in result
	result := executeTool("shell", map[string]any{"cmd": "echo err >&2"}, "/tmp")
	if !strings.Contains(result, "err") {
		t.Errorf("expected stderr in result, got %q", result)
	}
}

func TestReadWriteFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	content := "hello world"

	writeResult := executeTool("write_file", map[string]any{"path": path, "content": content}, dir)
	if !strings.Contains(writeResult, "Wrote") {
		t.Errorf("unexpected write result: %q", writeResult)
	}

	readResult := executeTool("read_file", map[string]any{"path": path}, dir)
	if readResult != content {
		t.Errorf("expected %q, got %q", content, readResult)
	}
}

func TestReadFileRelative(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "foo.txt"), []byte("bar"), 0644); err != nil {
		t.Fatal(err)
	}
	result := executeTool("read_file", map[string]any{"path": "foo.txt"}, dir)
	if result != "bar" {
		t.Errorf("expected 'bar', got %q", result)
	}
}

func TestWriteFileCreatesDir(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "subdir", "nested.txt")
	result := executeTool("write_file", map[string]any{"path": path, "content": "data"}, dir)
	if !strings.Contains(result, "Wrote") {
		t.Errorf("unexpected write result: %q", result)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("file not created: %v", err)
	}
}

func TestEditFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "edit.txt")
	os.WriteFile(path, []byte("hello world\nfoo bar\n"), 0644)

	result := executeTool("edit_file", map[string]any{
		"path": path, "old_str": "hello world", "new_str": "goodbye world",
	}, dir)
	if !strings.Contains(result, "Edited") {
		t.Errorf("expected Edited, got %q", result)
	}
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), "goodbye world") {
		t.Errorf("edit not applied: %q", string(data))
	}
	if strings.Contains(string(data), "hello world") {
		t.Errorf("old_str still present: %q", string(data))
	}
}

func TestEditFileNotFound(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "edit.txt")
	os.WriteFile(path, []byte("hello world\n"), 0644)

	result := executeTool("edit_file", map[string]any{
		"path": path, "old_str": "not present", "new_str": "x",
	}, dir)
	if !strings.Contains(result, "ERROR") {
		t.Errorf("expected ERROR for missing old_str, got %q", result)
	}
}

func TestListFiles(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.go"), []byte(""), 0644)
	os.WriteFile(filepath.Join(dir, "b.go"), []byte(""), 0644)
	os.WriteFile(filepath.Join(dir, "c.txt"), []byte(""), 0644)

	result := executeTool("list_files", map[string]any{"pattern": "*.go"}, dir)
	lines := strings.Split(strings.TrimSpace(result), "\n")
	if len(lines) != 2 {
		t.Errorf("expected 2 .go files, got %d: %q", len(lines), result)
	}
}

func TestListFilesDoublestar(t *testing.T) {
	dir := t.TempDir()
	subdir := filepath.Join(dir, "sub")
	os.Mkdir(subdir, 0755)
	os.WriteFile(filepath.Join(dir, "a.go"), []byte(""), 0644)
	os.WriteFile(filepath.Join(subdir, "b.go"), []byte(""), 0644)

	result := executeTool("list_files", map[string]any{"pattern": "**/*.go"}, dir)
	if !strings.Contains(result, "a.go") || !strings.Contains(result, "b.go") {
		t.Errorf("expected both .go files, got %q", result)
	}
}

func TestListFilesNoMatch(t *testing.T) {
	dir := t.TempDir()
	result := executeTool("list_files", map[string]any{"pattern": "*.xyz"}, dir)
	if result != "(no matches)" {
		t.Errorf("expected '(no matches)', got %q", result)
	}
}

func TestSearch(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "test.txt"), []byte("hello world\nfoo bar\n"), 0644)

	result := executeTool("search", map[string]any{"pattern": "hello", "path": "."}, dir)
	if !strings.Contains(result, "hello") {
		t.Errorf("expected 'hello' in result, got %q", result)
	}
}

func TestSearchNoMatch(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "test.txt"), []byte("nothing here\n"), 0644)

	result := executeTool("search", map[string]any{"pattern": "xyzzy123"}, dir)
	if result != "(no matches)" {
		t.Errorf("expected '(no matches)', got %q", result)
	}
}

func TestUnknownTool(t *testing.T) {
	result := executeTool("nonexistent", map[string]any{}, "/tmp")
	if !strings.Contains(result, "ERROR") {
		t.Errorf("expected ERROR, got %q", result)
	}
}

