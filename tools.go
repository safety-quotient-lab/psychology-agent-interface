package main

import (
	"os"

	plog "github.com/safety-quotient-lab/psychology-agent-interface/pkg/log"
	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/tool"
)

// defaultRegistry creates a tool registry with all standard tools.
// web_search included only when KAGI_API_KEY present.
func defaultRegistry() *tool.Registry {
	r := tool.NewRegistry()
	tool.RegisterBuiltins(r, os.Getenv("KAGI_API_KEY"))
	return r
}

// executeTool runs a named tool via the default registry.
// Compatibility wrapper — callers should migrate to registry.Run().
func executeTool(name string, args map[string]any, cwd string) string {
	plog.L.Debug("tool exec", "name", name)
	return defaultRegistry().Run(name, args, cwd)
}

// Package-level wrappers for tool utilities used by model.go.
var (
	isShellCommand = tool.IsShellCommand
	runDirectShell = tool.RunDirectShell
)
