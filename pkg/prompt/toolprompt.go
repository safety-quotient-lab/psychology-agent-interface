package prompt

import "os"

// StaticToolProvider implements ToolProvider with a fixed tool list and
// descriptions. The tool registry provides these strings at construction time.
type StaticToolProvider struct {
	List string // comma-separated names for native mode
	Desc string // full descriptions for ReAct mode
}

func (s StaticToolProvider) ToolList() string         { return s.List }
func (s StaticToolProvider) ToolDescriptions() string { return s.Desc }

// DefaultToolProvider returns a StaticToolProvider with the standard tool set.
// Includes web_search only when KAGI_API_KEY exists in the environment.
// Includes LSP tools when hasLSP true (gopls running).
func DefaultToolProvider(hasLSP ...bool) StaticToolProvider {
	hasKagi := os.Getenv("KAGI_API_KEY") != ""
	lsp := len(hasLSP) > 0 && hasLSP[0]

	list := "shell, read_file, write_file, edit_file, list_files, search, fetch_url, ask_user"
	if hasKagi {
		list = "shell, read_file, write_file, edit_file, list_files, search, fetch_url, web_search, ask_user"
	}
	if lsp {
		list += ", lsp_definition, lsp_references, lsp_hover, lsp_diagnostics"
	}

	webSearchDesc := ""
	if hasKagi {
		webSearchDesc = "\n- web_search(query, [limit]): Search the web via Kagi. Returns titles, URLs, snippets. Default limit 5."
	}

	lspDesc := ""
	if lsp {
		lspDesc = `
- lsp_definition(path, line, col): Jump to the definition of a Go symbol. Line and col are 1-based. Returns file:line:col.
- lsp_references(path, line, col): Find all references to a Go symbol. Returns file:line:col for each.
- lsp_hover(path, line, col): Show type signature and documentation for a Go symbol.
- lsp_diagnostics(path): Get compiler errors and warnings for a Go source file.`
	}

	desc := `You also have access to local tools for investigating files and running commands.

Available tools:
- shell(cmd): Execute a bash command. Returns combined stdout and stderr.
- read_file(path): Read the full contents of a file.
- write_file(path, content): Write content to a file (overwrites if it exists).
- edit_file(path, old_str, new_str): Replace the first occurrence of old_str with new_str. old_str must match exactly (whitespace matters). Prefer this over write_file for targeted edits.
- list_files(pattern): List files matching a glob pattern. Use "*.ext" for current dir, "**/*.ext" to recurse.
- search(pattern): Search for a regex pattern in files (like grep -rn).
- fetch_url(url): Fetch a URL and return its content as plain text. Use for docs, GitHub issues, paste links.` + webSearchDesc + `
- ask_user(question): Ask the user a clarifying question and get their answer.` + lspDesc + `

To call a tool, output EXACTLY this format on its own line (nothing else on that line):
TOOL_CALL: {"name": "tool_name", "arguments": {"arg": "value"}}

IMPORTANT: After writing a TOOL_CALL line, STOP IMMEDIATELY. Do not write anything else.
The system will execute the tool and provide the result. Never fabricate tool results.
When finished, write your final answer with no TOOL_CALL lines.`

	return StaticToolProvider{List: list, Desc: desc}
}
