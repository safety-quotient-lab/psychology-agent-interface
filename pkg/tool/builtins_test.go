package tool

import (
	"strings"
	"testing"
)

func TestTruncate(t *testing.T) {
	long := strings.Repeat("a", 5000)
	result := truncate(long)
	if !strings.Contains(result, "[truncated]") {
		t.Errorf("expected truncation marker")
	}
	if len(result) > truncLimit+50 {
		t.Errorf("result too long: %d", len(result))
	}
}

func TestTruncateShort(t *testing.T) {
	s := "short string"
	result := truncate(s)
	if result != s {
		t.Errorf("short string should not be truncated, got %q", result)
	}
}
