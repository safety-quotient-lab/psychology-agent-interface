package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// Skill is a named, invocable capability that configures the agent's analytical mode
// via a specialized system prompt overlay. Modelled after the A2A skill declarations
// in the unratified interagent mesh (/.well-known/agent-card.json).
type Skill struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	SystemPrompt string   `json:"system_prompt"` // appended to base system prompt when active
	Tags         []string `json:"tags,omitempty"`
	Examples     []string `json:"examples,omitempty"`
}

// loadSkills merges skills from global config and project-local config.
// Project-local skills with the same ID override global ones.
//
// Config locations (loaded in order, later overrides earlier):
//
//	~/.config/psyai/skills.json   — global skills
//	<cwd>/psyai-skills.json        — project-local skills
func loadSkills(cwd string) []Skill {
	home, _ := os.UserHomeDir()
	paths := []string{
		filepath.Join(home, ".config", "psyai", "skills.json"),
		filepath.Join(cwd, "psyai-skills.json"),
	}
	var skills []Skill
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var loaded []Skill
		if err := json.Unmarshal(data, &loaded); err != nil {
			continue
		}
		for _, s := range loaded {
			replaced := false
			for i, existing := range skills {
				if existing.ID == s.ID {
					skills[i] = s
					replaced = true
					break
				}
			}
			if !replaced {
				skills = append(skills, s)
			}
		}
	}
	return skills
}

// findSkill returns the skill with the given ID, or nil.
func findSkill(skills []Skill, id string) *Skill {
	for i := range skills {
		if skills[i].ID == id {
			return &skills[i]
		}
	}
	return nil
}
