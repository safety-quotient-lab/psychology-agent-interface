// Package tool provides a data-driven tool registry that replaces the monolithic
// executeTool switch statement. Each tool self-describes its name, parameters,
// and executor function (Plan 9 Phase 1).
package tool

import (
	"fmt"
	"sort"
)

// Param describes a tool parameter.
type Param struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
	Required    bool   `json:"required"`
	Default     string `json:"default,omitempty"`
}

// Tool describes a callable tool with its metadata and executor.
type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  []Param `json:"parameters"`
	// Execute runs the tool with the given args and working directory.
	// Not serialized — wired at registration time.
	Execute func(args map[string]any, cwd string) string `json:"-"`
}

// Registry holds available tools, keyed by name.
type Registry struct {
	tools map[string]Tool
	order []string // insertion order for stable iteration
}

// NewRegistry creates an empty tool registry.
func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Tool)}
}

// Register adds a tool to the registry.
func (r *Registry) Register(t Tool) {
	if _, exists := r.tools[t.Name]; !exists {
		r.order = append(r.order, t.Name)
	}
	r.tools[t.Name] = t
}

// Get returns a tool by name and whether it exists.
func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.tools[name]
	return t, ok
}

// Run executes a named tool. Returns an error string for unknown tools.
func (r *Registry) Run(name string, args map[string]any, cwd string) string {
	t, ok := r.tools[name]
	if !ok {
		return fmt.Sprintf("ERROR: unknown tool '%s'", name)
	}
	return t.Execute(args, cwd)
}

// Names returns tool names in registration order.
func (r *Registry) Names() []string {
	out := make([]string, len(r.order))
	copy(out, r.order)
	return out
}

// SortedNames returns tool names sorted alphabetically.
func (r *Registry) SortedNames() []string {
	names := r.Names()
	sort.Strings(names)
	return names
}

// ToolList returns a comma-separated list of tool names for system prompts.
func (r *Registry) ToolList() string {
	result := ""
	for i, name := range r.order {
		if i > 0 {
			result += ", "
		}
		result += name
	}
	return result
}

// Count returns the number of registered tools.
func (r *Registry) Count() int {
	return len(r.tools)
}

// Remove unregisters a tool by name.
func (r *Registry) Remove(name string) {
	delete(r.tools, name)
	for i, n := range r.order {
		if n == name {
			r.order = append(r.order[:i], r.order[i+1:]...)
			break
		}
	}
}
