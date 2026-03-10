// Package config handles environment, .env, and .dev.vars loading.
package config

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

// LoadDotFile reads a KEY=VALUE file into os environment.
func LoadDotFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		eq := strings.IndexByte(line, '=')
		if eq <= 0 {
			continue
		}
		key := strings.TrimSpace(line[:eq])
		val := strings.TrimSpace(line[eq+1:])
		val = strings.Trim(val, `"'`)
		os.Setenv(key, val)
	}
	return sc.Err()
}

// ProjectRoot walks up from cwd to find the directory containing go.mod or package.json.
func ProjectRoot() string {
	dir, _ := os.Getwd()
	for {
		for _, marker := range []string{"go.mod", "package.json"} {
			if _, err := os.Stat(filepath.Join(dir, marker)); err == nil {
				return dir
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	// Fallback to cwd
	cwd, _ := os.Getwd()
	return cwd
}

// LoadProjectEnv loads .env and .dev.vars from the project root.
func LoadProjectEnv() {
	root := ProjectRoot()
	LoadDotFile(filepath.Join(root, ".env"))
	LoadDotFile(filepath.Join(root, ".dev.vars"))
}
