package prompt

// WorkspaceContext implements ContextProvider with mutable state
// for the current working directory, file list, and project instructions.
type WorkspaceContext struct {
	CWD      string
	Files    string // file listing (from list_files tool)
	ClaudeMD string // project instructions from CLAUDE.md
}

func (w *WorkspaceContext) WorkingDir() string          { return w.CWD }
func (w *WorkspaceContext) FileList() string             { return w.Files }
func (w *WorkspaceContext) ProjectInstructions() string  { return w.ClaudeMD }
