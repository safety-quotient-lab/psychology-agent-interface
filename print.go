package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/safety-quotient-lab/psychology-agent-interface/pkg/prompt"
)

// runPrint is the headless --print mode equivalent of claude -p.
// It runs a single prompt to completion (with tool calls) and writes the
// final reply to stdout. Suitable for scripting and piping.
func runPrint(c appConfig, proc inferProc) error {
	// Resolve prompt: from positional args, or stdin if piped.
	userPrompt := strings.TrimSpace(c.prompt)
	if userPrompt == "" {
		stat, _ := os.Stdin.Stat()
		if (stat.Mode() & os.ModeCharDevice) == 0 {
			// stdin is a pipe
			b, err := io.ReadAll(os.Stdin)
			if err != nil {
				return fmt.Errorf("reading stdin: %w", err)
			}
			userPrompt = strings.TrimSpace(string(b))
		}
	}
	if userPrompt == "" {
		return fmt.Errorf("no prompt: pass text after flags or pipe via stdin")
	}

	// Wait for backend to be ready.
	rp, err := proc.waitReady()
	if err != nil {
		return fmt.Errorf("backend not ready: %w", err)
	}
	useNative := rp.UseNative
	tier := c.catalog.Tier(c.model)
	t := Turn{Native: useNative, CWD: c.cwd}

	// Build namespace and system prompt with file listing embedded.
	fileList := executeTool("list_files", map[string]any{"pattern": "*"}, c.cwd)
	ns := prompt.Namespace{
		Identity: prompt.PsychologyIdentity{},
		Tools:    prompt.DefaultToolProvider(),
		Context:  &prompt.WorkspaceContext{CWD: c.cwd, Files: fileList},
		Format:   prompt.PsychologyFormat{},
		FewShot:  prompt.PsychologyFewShot{},
	}
	sysprompt, priming := ns.Build(tier, useNative)

	msgs := []Message{{Role: "system", Content: sysprompt}}
	msgs = append(msgs, promptMsgsToMain(priming)...)
	msgs = append(msgs, Message{Role: "user", Content: userPrompt})

	maxTurns := c.maxTurns
	if maxTurns <= 0 {
		maxTurns = 15
	}

	// nudge wraps msgs with per-turn format reinforcement.
	nudge := func(conv []Message) []Message {
		nudged := ns.WithNudge(mainMsgsToPrompt(conv), tier)
		return promptMsgsToMain(nudged)
	}

	lp, canStream := proc.(*llmProc)

	for turn := 0; turn < maxTurns; turn++ {
		// Stream the final answer if requested; otherwise use blocking inference.
		// Tool-call turns always use blocking so we can parse the full reply.
		if c.stream && canStream {
			var buf strings.Builder
			_, _, err := lp.inferStream(nudge(msgs), 1024, 0.2, &buf)
			if err != nil {
				return fmt.Errorf("inference: %w", err)
			}
			reply := buf.String()

			results, hadTools := t.ProcessReply(reply)
			if !hadTools {
				// Final answer — re-run streaming to stdout.
				_, _, err = lp.inferStream(nudge(msgs), 1024, 0.2, os.Stdout)
				if err != nil {
					return fmt.Errorf("inference (stream): %w", err)
				}
				fmt.Println()
				return nil
			}

			msgs = append(msgs, Message{Role: "assistant", Content: reply})
			msgs = append(msgs, results...)
			continue
		}

		// Blocking inference (default).
		ir, err := proc.infer(nudge(msgs), 1024, 0.2)
		if err != nil {
			return fmt.Errorf("inference: %w", err)
		}
		if ir.Error != "" {
			return fmt.Errorf("inference: %s", ir.Error)
		}
		reply := ir.Reply

		results, hadTools := t.ProcessReply(reply)
		if !hadTools {
			fmt.Println(stripMarkup(reply))
			return nil
		}

		msgs = append(msgs, Message{Role: "assistant", Content: reply})
		msgs = append(msgs, results...)
	}

	return fmt.Errorf("reached max-turns (%d) without a final answer", maxTurns)
}
