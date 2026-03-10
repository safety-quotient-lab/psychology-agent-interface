package log

import (
	"os"
	"path/filepath"
	"strings"

	clog "github.com/charmbracelet/log"
)

// L is the package-level logger instance.
var L *clog.Logger

func init() {
	// Default to file-based logging to avoid polluting TUI stdout/stderr.
	logDir := filepath.Join(homeDir(), ".local", "state", "pai")
	os.MkdirAll(logDir, 0755)
	logPath := filepath.Join(logDir, "debug.log")
	f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		// Fallback: discard
		f, _ = os.Open(os.DevNull)
	}

	L = clog.NewWithOptions(f, clog.Options{
		ReportTimestamp: true,
		ReportCaller:   true,
	})

	// Set level from PAI_LOG_LEVEL env var
	switch strings.ToLower(os.Getenv("PAI_LOG_LEVEL")) {
	case "debug":
		L.SetLevel(clog.DebugLevel)
	case "info":
		L.SetLevel(clog.InfoLevel)
	case "warn", "":
		L.SetLevel(clog.WarnLevel)
	case "error":
		L.SetLevel(clog.ErrorLevel)
	}
}

func homeDir() string {
	h, _ := os.UserHomeDir()
	return h
}
