package tool

import (
	"fmt"
	"path/filepath"

	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/lsp"
)

// RegisterLSP adds Go language server tools to the registry.
// The LSP client manages a gopls subprocess for the given workspace root.
// Pass client=nil to skip registration (e.g., when gopls unavailable).
func RegisterLSP(r *Registry, client *lsp.Client) {
	if client == nil {
		return
	}

	r.Register(Tool{
		Name:        "lsp_definition",
		Description: "Jump to the definition of a Go symbol at a given file position. Returns file:line:col.",
		Parameters: []Param{
			{Name: "path", Type: "string", Description: "Go source file path", Required: true},
			{Name: "line", Type: "integer", Description: "Line number (1-based)", Required: true},
			{Name: "col", Type: "integer", Description: "Column number (1-based)", Required: true},
		},
		Execute: func(args map[string]any, cwd string) string {
			path, line, col, err := parseLSPArgs(args, cwd)
			if err != "" {
				return err
			}
			result, e := client.Definition(path, line, col)
			if e != nil {
				return fmt.Sprintf("ERROR: %v", e)
			}
			return result
		},
	})

	r.Register(Tool{
		Name:        "lsp_references",
		Description: "Find all references to a Go symbol at a given file position. Returns file:line:col for each reference.",
		Parameters: []Param{
			{Name: "path", Type: "string", Description: "Go source file path", Required: true},
			{Name: "line", Type: "integer", Description: "Line number (1-based)", Required: true},
			{Name: "col", Type: "integer", Description: "Column number (1-based)", Required: true},
		},
		Execute: func(args map[string]any, cwd string) string {
			path, line, col, err := parseLSPArgs(args, cwd)
			if err != "" {
				return err
			}
			result, e := client.References(path, line, col)
			if e != nil {
				return fmt.Sprintf("ERROR: %v", e)
			}
			return result
		},
	})

	r.Register(Tool{
		Name:        "lsp_hover",
		Description: "Show type signature and documentation for a Go symbol at a given file position.",
		Parameters: []Param{
			{Name: "path", Type: "string", Description: "Go source file path", Required: true},
			{Name: "line", Type: "integer", Description: "Line number (1-based)", Required: true},
			{Name: "col", Type: "integer", Description: "Column number (1-based)", Required: true},
		},
		Execute: func(args map[string]any, cwd string) string {
			path, line, col, err := parseLSPArgs(args, cwd)
			if err != "" {
				return err
			}
			result, e := client.Hover(path, line, col)
			if e != nil {
				return fmt.Sprintf("ERROR: %v", e)
			}
			return result
		},
	})

	r.Register(Tool{
		Name:        "lsp_diagnostics",
		Description: "Get compiler errors and warnings for a Go source file.",
		Parameters: []Param{
			{Name: "path", Type: "string", Description: "Go source file path", Required: true},
		},
		Execute: func(args map[string]any, cwd string) string {
			path, _ := args["path"].(string)
			if path == "" {
				return "ERROR: path required"
			}
			if !filepath.IsAbs(path) {
				path = filepath.Join(cwd, path)
			}
			result, e := client.Diagnostics(path)
			if e != nil {
				return fmt.Sprintf("ERROR: %v", e)
			}
			return result
		},
	})
}

// parseLSPArgs extracts path, line, col from tool arguments.
// Returns an error string (empty on success).
func parseLSPArgs(args map[string]any, cwd string) (string, int, int, string) {
	path, _ := args["path"].(string)
	if path == "" {
		return "", 0, 0, "ERROR: path required"
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(cwd, path)
	}

	lineF, ok := args["line"].(float64)
	if !ok || lineF < 1 {
		return "", 0, 0, "ERROR: line required (1-based integer)"
	}
	colF, ok := args["col"].(float64)
	if !ok || colF < 1 {
		return "", 0, 0, "ERROR: col required (1-based integer)"
	}
	return path, int(lineF), int(colF), ""
}
