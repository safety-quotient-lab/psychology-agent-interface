package main

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// runPrint is the headless --print mode equivalent of claude -p.
// It runs a single prompt to completion (with tool calls) and writes the
// final reply to stdout. Suitable for scripting and piping.
func runPrint(c appConfig, proc inferProc) error {
	// Resolve prompt: from positional args, or stdin if piped.
	prompt := strings.TrimSpace(c.prompt)
	if prompt == "" {
		stat, _ := os.Stdin.Stat()
		if (stat.Mode() & os.ModeCharDevice) == 0 {
			// stdin is a pipe
			b, err := io.ReadAll(os.Stdin)
			if err != nil {
				return fmt.Errorf("reading stdin: %w", err)
			}
			prompt = strings.TrimSpace(string(b))
		}
	}
	if prompt == "" {
		return fmt.Errorf("no prompt: pass text after flags or pipe via stdin")
	}

	// Wait for backend to be ready.
	rp, err := proc.waitReady()
	if err != nil {
		return fmt.Errorf("backend not ready: %w", err)
	}
	useNative := rp.UseNative

	// Build system prompt with file listing.
	fileList := executeTool("list_files", map[string]any{"pattern": "*"}, c.cwd)
	var sysprompt string
	if useNative {
		sysprompt = nativeSystem(c.cwd, fileList, "")
	} else {
		sysprompt = reactSystem(c.cwd, fileList, "")
	}

	// Prime the conversation with two fake prior tool-use exchanges so small models
	// see concrete examples of tool call format before they need to emit one.
	// list_files first (static), then shell("pwd") → actual cwd.
	var primed []Message
	if useNative {
		primed = []Message{
			{Role: "assistant", Content: `<tool_call>{"name":"list_files","arguments":{"pattern":"*"}}</tool_call>`},
			{Role: "tool", Name: "list_files", Content: fileList},
			{Role: "assistant", Content: `<tool_call>{"name":"shell","arguments":{"cmd":"pwd"}}</tool_call>`},
			{Role: "tool", Name: "shell", Content: c.cwd},
		}
	} else {
		primed = []Message{
			{Role: "assistant", Content: `TOOL_CALL: {"name": "list_files", "arguments": {"pattern": "*"}}`},
			{Role: "user", Content: "TOOL_RESULT (list_files):\n" + fileList + "\n\nContinue."},
			{Role: "assistant", Content: `TOOL_CALL: {"name": "shell", "arguments": {"cmd": "pwd"}}`},
			{Role: "user", Content: "TOOL_RESULT (shell):\n" + c.cwd + "\n\nContinue."},
		}
	}

	msgs := append([]Message{{Role: "system", Content: sysprompt}}, primed...)
	msgs = append(msgs, Message{Role: "user", Content: prompt})

	maxTurns := c.maxTurns
	if maxTurns <= 0 {
		maxTurns = 15
	}

	lp, canStream := proc.(*llmProc)

	for turn := 0; turn < maxTurns; turn++ {
		// Stream the final answer if requested; otherwise use blocking inference.
		// Tool-call turns always use blocking so we can parse the full reply.
		if c.stream && canStream {
			// Use streaming inference. Tokens for tool-call turns are buffered
			// (not written to stdout). Tokens for the final answer go to stdout.
			var buf strings.Builder
			_, _, err := lp.inferStream(msgs, 1024, 0.2, &buf)
			if err != nil {
				return fmt.Errorf("inference: %w", err)
			}
			reply := buf.String()

			var calls []ToolCall
			if useNative {
				calls = parseNative(reply)
			} else {
				calls = parseReact(reply)
			}

			if len(calls) == 0 {
				// Final answer — re-run as streaming so the user sees tokens live.
				// The model is already primed; this second call should reproduce
				// the same reply (temperature 0.2 is low but not zero, so not
				// guaranteed identical — acceptable trade-off vs. buffering).
				_, _, err = lp.inferStream(msgs, 1024, 0.2, os.Stdout)
				if err != nil {
					return fmt.Errorf("inference (stream): %w", err)
				}
				fmt.Println()
				return nil
			}

			msgs = append(msgs, Message{Role: "assistant", Content: reply})
			for _, call := range calls {
				result := executeTool(call.Name, call.Args, c.cwd)
				if useNative {
					msgs = append(msgs, Message{Role: "tool", Name: call.Name, Content: result})
				} else {
					msgs = append(msgs, Message{
						Role:    "user",
						Content: fmt.Sprintf("TOOL_RESULT (%s):\n%s\n\nContinue.", call.Name, result),
					})
				}
			}
			continue
		}

		// Blocking inference (default).
		ir, err := proc.infer(msgs, 1024, 0.2)
		if err != nil {
			return fmt.Errorf("inference: %w", err)
		}
		if ir.Error != "" {
			return fmt.Errorf("inference: %s", ir.Error)
		}
		reply := ir.Reply

		var calls []ToolCall
		if useNative {
			calls = parseNative(reply)
		} else {
			calls = parseReact(reply)
		}

		if len(calls) == 0 {
			fmt.Println(stripMarkup(reply))
			return nil
		}

		msgs = append(msgs, Message{Role: "assistant", Content: reply})
		for _, call := range calls {
			result := executeTool(call.Name, call.Args, c.cwd)
			if useNative {
				msgs = append(msgs, Message{Role: "tool", Name: call.Name, Content: result})
			} else {
				msgs = append(msgs, Message{
					Role:    "user",
					Content: fmt.Sprintf("TOOL_RESULT (%s):\n%s\n\nContinue.", call.Name, result),
				})
			}
		}
	}

	return fmt.Errorf("reached max-turns (%d) without a final answer", maxTurns)
}
