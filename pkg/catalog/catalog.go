// Package catalog provides a data-driven model catalog loaded from models.json.
// Replaces scattered model metadata across Go and Python code with a single
// source of truth (Plan 9 Phase 5).
package catalog

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// ModelSpec describes a model's full metadata. Loaded from models.json.
type ModelSpec struct {
	Key             string `json:"key"`
	Label           string `json:"label"`
	HuggingFace     string `json:"huggingface"`
	Tier            int    `json:"tier"`
	ContextLimit    int    `json:"context_limit"`
	DeclaredCtx     string `json:"declared_ctx"`
	VRAM            string `json:"vram"`
	Dtype           string `json:"dtype"`
	Gated           bool   `json:"gated"`
	NativeTools     bool   `json:"native_tools"`
	MPSNativeStable bool   `json:"mps_native_stable"`
	Notes           string `json:"notes"`
}

// DisplayNote returns the note string formatted for the model selector UI.
// Combines gated status, special markers, and user-facing notes.
func (m ModelSpec) DisplayNote() string {
	parts := []string{}
	if m.Key == "qwen-0.5b" {
		parts = append(parts, "\u2605 fastest")
	}
	if m.Gated {
		parts = append(parts, "gated")
	}
	if m.Notes == "heavy" {
		parts = append(parts, "\u26a0 heavy")
	}
	if len(parts) == 0 {
		return ""
	}
	result := parts[0]
	for _, p := range parts[1:] {
		result += " " + p
	}
	return result
}

// TierLabel returns "Tier 1", "Tier 2", or "Tier 3".
func (m ModelSpec) TierLabel() string {
	return fmt.Sprintf("Tier %d", m.Tier)
}

// Catalog holds all known models.
type Catalog struct {
	models []ModelSpec
	byKey  map[string]*ModelSpec
}

// Load reads models.json from the given project root directory.
func Load(projectRoot string) (*Catalog, error) {
	path := filepath.Join(projectRoot, "models.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load model catalog: %w", err)
	}
	var specs []ModelSpec
	if err := json.Unmarshal(data, &specs); err != nil {
		return nil, fmt.Errorf("parse model catalog: %w", err)
	}
	c := &Catalog{
		models: specs,
		byKey:  make(map[string]*ModelSpec, len(specs)),
	}
	for i := range c.models {
		c.byKey[c.models[i].Key] = &c.models[i]
	}
	return c, nil
}

// Get returns the spec for a model key, or nil if unknown.
func (c *Catalog) Get(key string) *ModelSpec {
	return c.byKey[key]
}

// Tier returns the tier number for a model (1, 2, or 3). Returns 1 for unknown.
func (c *Catalog) Tier(key string) int {
	if m := c.byKey[key]; m != nil {
		return m.Tier
	}
	return 1
}

// ContextLimit returns the conservative effective token limit. Returns 4096 for unknown.
func (c *Catalog) ContextLimit(key string) int {
	if m := c.byKey[key]; m != nil {
		return m.ContextLimit
	}
	return 4096
}

// All returns all model specs in catalog order.
func (c *Catalog) All() []ModelSpec {
	return c.models
}

// ValidKeys returns sorted model keys for CLI validation.
func (c *Catalog) ValidKeys() []string {
	keys := make([]string, len(c.models))
	for i, m := range c.models {
		keys[i] = m.Key
	}
	sort.Strings(keys)
	return keys
}

// RecommendedTurns returns the approximate number of user-assistant exchanges
// before context quality degrades. Derived from context_limit minus estimated
// system prompt overhead (~1200 tokens for tier 1, ~1500 for tier 2+) and
// token reserve (1200), divided by ~200 tokens per turn (user + assistant).
// Returns 0 for unknown models (no limit displayed).
func (c *Catalog) RecommendedTurns(key string) int {
	m := c.byKey[key]
	if m == nil {
		return 0
	}
	// Estimate system prompt size by tier.
	promptOverhead := 1500
	if m.Tier <= 1 {
		promptOverhead = 1200 // slim tool descriptions
	}
	available := m.ContextLimit - 1200 - promptOverhead // 1200 = tokenReserve
	if available <= 0 {
		return 1
	}
	return available / 200 // ~100 tok user + ~100 tok response
}

// IsTier1 returns true if the model belongs to Tier 1 (≤2B params).
func (c *Catalog) IsTier1(key string) bool { return c.Tier(key) == 1 }

// IsTier2 returns true if the model belongs to Tier 2 (2B-4B params).
func (c *Catalog) IsTier2(key string) bool { return c.Tier(key) == 2 }
