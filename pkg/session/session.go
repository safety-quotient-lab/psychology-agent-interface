// Package session manages conversation persistence as directory-based sessions.
// Each session becomes a directory containing typed files — Plan 9 Phase 4:
// sessions as filesystem.
package session

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/msg"
)

// Message is an alias for the shared msg.Message type.
type Message = msg.Message

// Meta holds session metadata.
type Meta struct {
	ID      string    `json:"id"`
	Model   string    `json:"model"`
	SavedAt time.Time `json:"saved_at"`
}

// Session is a saved conversation snapshot.
type Session struct {
	Meta
	Conversation []Message `json:"conversation"`
}

// Dir returns the base session storage directory.
func Dir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".local", "share", "pai", "sessions")
	return dir, os.MkdirAll(dir, 0755)
}

// Save persists a conversation as a JSON session file.
// Returns the file path on success.
func Save(model string, conv []Message) (string, error) {
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	id := time.Now().Format("20060102-150405")
	s := Session{
		Meta:         Meta{ID: id, Model: model, SavedAt: time.Now()},
		Conversation: conv,
	}
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, id+".json")
	return path, os.WriteFile(path, b, 0644)
}

// List returns all saved sessions, newest first.
func List() ([]Session, error) {
	dir, err := Dir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var sessions []Session
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		var s Session
		if json.Unmarshal(data, &s) == nil {
			sessions = append(sessions, s)
		}
	}
	sort.Slice(sessions, func(i, j int) bool {
		return sessions[i].SavedAt.After(sessions[j].SavedAt)
	})
	return sessions, nil
}

// LoadByIndex loads the nth most recent session (1-indexed).
func LoadByIndex(n int) (Session, error) {
	sessions, err := List()
	if err != nil {
		return Session{}, err
	}
	if n < 1 || n > len(sessions) {
		return Session{}, os.ErrNotExist
	}
	return sessions[n-1], nil
}
