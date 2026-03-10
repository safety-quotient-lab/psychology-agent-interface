// Package prompt assembles system prompts from independent provider resources.
// Each provider contributes one concern (identity, tools, context, format,
// few-shot examples). The Namespace composes them into a complete prompt —
// Plan 9 Phase 2: per-process namespaces.
package prompt

import "strings"

// Message mirrors the main package's Message type.
// Defined here so prompt stays dependency-free.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

// Namespace assembles a complete system prompt from independent resources.
// Each resource provides content without knowing about the others.
type Namespace struct {
	Identity IdentityProvider
	Tools    ToolProvider
	Context  ContextProvider
	Format   FormatProvider
	FewShot  FewShotProvider
	Skill    *SkillOverlay // nil when no skill active
}

// SkillOverlay holds an active skill's contribution to the system prompt.
type SkillOverlay struct {
	Name         string
	SystemPrompt string
}

// Build composes the final system prompt and priming messages.
func (ns Namespace) Build(tier int, native bool) (systemPrompt string, priming []Message) {
	var sb strings.Builder

	// 1. Identity (who)
	if ns.Identity != nil {
		sb.WriteString(ns.Identity.Prompt(tier))
	}

	// 2. Working directory
	if ns.Context != nil {
		cwd := ns.Context.WorkingDir()
		if cwd != "" {
			sb.WriteString("\n\nWorking directory: ")
			sb.WriteString(cwd)
		}
	}

	// 3. Tools (what)
	if ns.Tools != nil {
		if native {
			list := ns.Tools.ToolList()
			if list != "" {
				sb.WriteString("\nYou have tools: ")
				sb.WriteString(list)
				sb.WriteString(".\nUse tools when you need to investigate files or run commands.")
			}
		} else {
			desc := ns.Tools.ToolDescriptions()
			if desc != "" {
				sb.WriteString("\n\n")
				sb.WriteString(desc)
			}
		}
	}

	// 4. Context (where) — file list, project instructions
	if ns.Context != nil {
		fl := ns.Context.FileList()
		if fl != "" {
			sb.WriteString("\n\nFiles in working directory:\n")
			sb.WriteString(fl)
		}
		pi := ns.Context.ProjectInstructions()
		if pi != "" {
			sb.WriteString("\n\n# Project instructions\n")
			sb.WriteString(pi)
		}
	}

	// 5. Skill overlay
	if ns.Skill != nil {
		sb.WriteString("\n\n# Active skill: ")
		sb.WriteString(ns.Skill.Name)
		sb.WriteString("\n")
		sb.WriteString(ns.Skill.SystemPrompt)
	}

	// 6. Format rules (how) — last, closest to conversation
	if ns.Format != nil {
		rules := ns.Format.Rules(tier)
		if rules != "" {
			sb.WriteString(rules)
		}
	}

	systemPrompt = sb.String()

	// 7. Few-shot examples
	if ns.FewShot != nil {
		priming = ns.FewShot.Examples(tier)
	}

	return systemPrompt, priming
}

// NudgeMessages returns the per-turn format reinforcement as a slice.
// Returns nil when no nudge applies (Tier 3+).
func (ns Namespace) NudgeMessages(tier int) []Message {
	if ns.Format == nil {
		return nil
	}
	nudge := ns.Format.Nudge(tier)
	if nudge == "" {
		return nil
	}
	return []Message{{Role: "user", Content: strings.TrimSpace(nudge)}}
}

// WithNudge returns a copy of conv with format nudge appended (if any).
// The original slice remains unmodified.
func (ns Namespace) WithNudge(conv []Message, tier int) []Message {
	nudge := ns.NudgeMessages(tier)
	if len(nudge) == 0 {
		return conv
	}
	out := make([]Message, len(conv), len(conv)+len(nudge))
	copy(out, conv)
	return append(out, nudge...)
}
