package prompt

import (
	"strings"
	"testing"
)

func TestNamespaceBuildNative(t *testing.T) {
	ns := Namespace{
		Identity: PsychologyIdentity{},
		Tools:    StaticToolProvider{List: "shell, read_file", Desc: "tools desc"},
		Context:  &WorkspaceContext{CWD: "/tmp/test", Files: "a.go\nb.go"},
		Format:   PsychologyFormat{},
		FewShot:  PsychologyFewShot{},
	}
	sys, priming := ns.Build(1, true)
	if !strings.Contains(sys, "psychology research assistant") {
		t.Error("expected identity in system prompt")
	}
	if !strings.Contains(sys, "shell, read_file") {
		t.Error("expected tool list in native mode")
	}
	if !strings.Contains(sys, "/tmp/test") {
		t.Error("expected working directory")
	}
	if !strings.Contains(sys, "a.go") {
		t.Error("expected file listing")
	}
	if !strings.Contains(sys, "Reminder:") {
		t.Error("expected format reminder for tier 1")
	}
	if len(priming) != 2 {
		t.Errorf("expected 2 priming messages, got %d", len(priming))
	}
}

func TestNamespaceBuildReact(t *testing.T) {
	ns := Namespace{
		Identity: PsychologyIdentity{},
		Tools:    StaticToolProvider{List: "shell", Desc: "Available tools:\n- shell(cmd): run cmd"},
		Context:  &WorkspaceContext{CWD: "/tmp"},
		Format:   PsychologyFormat{},
	}
	sys, _ := ns.Build(2, false)
	if !strings.Contains(sys, "Available tools") {
		t.Error("expected full tool descriptions in ReAct mode")
	}
	if !strings.Contains(sys, "[OBS]") {
		t.Error("expected tier 2 identity")
	}
}

func TestNamespaceTier3NoNudge(t *testing.T) {
	ns := Namespace{Format: PsychologyFormat{}}
	nudge := ns.NudgeMessages(3)
	if len(nudge) != 0 {
		t.Error("tier 3 should produce no nudge")
	}
}

func TestNamespaceWithNudge(t *testing.T) {
	ns := Namespace{Format: PsychologyFormat{}}
	conv := []Message{{Role: "user", Content: "hello"}}
	nudged := ns.WithNudge(conv, 1)
	if len(nudged) != 2 {
		t.Errorf("expected 2 messages, got %d", len(nudged))
	}
	if nudged[1].Role != "user" {
		t.Error("nudge should appear as user message")
	}
	// Original unmodified
	if len(conv) != 1 {
		t.Error("original conversation modified")
	}
}

func TestSkillOverlay(t *testing.T) {
	ns := Namespace{
		Identity: PsychologyIdentity{},
		Skill:    &SkillOverlay{Name: "test-skill", SystemPrompt: "do the thing"},
	}
	sys, _ := ns.Build(1, true)
	if !strings.Contains(sys, "Active skill: test-skill") {
		t.Error("expected skill overlay in system prompt")
	}
	if !strings.Contains(sys, "do the thing") {
		t.Error("expected skill system prompt content")
	}
}

func TestCriticPrompt(t *testing.T) {
	cp := CriticPrompt("user said X", "assistant said Y")
	if !strings.Contains(cp, "Socratic") {
		t.Error("expected Socratic in critic prompt")
	}
	if !strings.Contains(cp, "user said X") {
		t.Error("expected user message")
	}
	if !strings.Contains(cp, "assistant said Y") {
		t.Error("expected assistant reply")
	}
}
