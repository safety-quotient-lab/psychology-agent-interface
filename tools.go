package main

import (
	"os"

	plog "github.com/safety-quotient-lab/psychology-agent-interface/pkg/log"
	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/lsp"
	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/tool"
)

// shared LSP client — initialized once, reused across tool calls.
var sharedLSPClient *lsp.Client

// initLSP starts gopls for the given workspace root.
// Call once at startup; safe to skip if gopls unavailable.
func initLSP(workspaceRoot string) {
	client, err := lsp.NewClient(workspaceRoot)
	if err != nil {
		plog.L.Info("LSP unavailable", "error", err)
		return
	}
	sharedLSPClient = client
	plog.L.Info("LSP started", "root", workspaceRoot)
}

// closeLSP shuts down the gopls subprocess if running.
func closeLSP() {
	if sharedLSPClient != nil {
		sharedLSPClient.Close()
		sharedLSPClient = nil
	}
}

// cachedRegistry holds the memoized tool registry.
// Built once on first access; avoids re-registering 13+ tools per call.
var cachedRegistry *tool.Registry

// defaultRegistry returns the shared tool registry, building it on first call.
// web_search included only when KAGI_API_KEY present.
// LSP tools included only when gopls running.
func defaultRegistry() *tool.Registry {
	if cachedRegistry == nil {
		r := tool.NewRegistry()
		tool.RegisterBuiltins(r, os.Getenv("KAGI_API_KEY"))
		tool.RegisterLSP(r, sharedLSPClient)
		cachedRegistry = r
	}
	return cachedRegistry
}

// executeTool runs a named tool via the shared registry.
func executeTool(name string, args map[string]any, cwd string) string {
	plog.L.Debug("tool exec", "name", name)
	return defaultRegistry().Run(name, args, cwd)
}

// Package-level wrappers for tool utilities used by model.go.
var (
	isShellCommand = tool.IsShellCommand
	runDirectShell = tool.RunDirectShell
)
