package main

import (
	"net/http"
	"testing"
	"time"
)

// TestGatewayHTTPProc exercises httpProc against the live gateway on :7705.
// Skipped if the gateway is not reachable (CI / offline).
func TestGatewayHTTPProc(t *testing.T) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://127.0.0.1:7705/health")
	if err != nil || resp.StatusCode != 200 {
		t.Skip("gateway not reachable on :7705 — skipping live test")
	}
	resp.Body.Close()

	proc := &httpProc{
		baseURL: "http://127.0.0.1:7705",
		model:   "qwen-0.5b",
		client:  &http.Client{Timeout: 60 * time.Second},
	}

	t.Run("waitReady", func(t *testing.T) {
		rp, err := proc.waitReady()
		if err != nil {
			t.Fatalf("waitReady: %v", err)
		}
		if rp.Status != "ready" {
			t.Errorf("expected status=ready, got %q", rp.Status)
		}
		if rp.Model != "qwen-0.5b" {
			t.Errorf("expected model=qwen-0.5b, got %q", rp.Model)
		}
		if !rp.UseNative {
			t.Errorf("expected UseNative=true for qwen model")
		}
		t.Logf("ready: model=%s useNative=%v", rp.Model, rp.UseNative)
	})

	t.Run("doInfer", func(t *testing.T) {
		msgs := []Message{{Role: "user", Content: "Reply with exactly one word: pong"}}
		ir, err := proc.doInfer(msgs, 10, 0)
		if err != nil {
			t.Fatalf("doInfer: %v", err)
		}
		if ir.Reply == "" {
			t.Error("empty reply")
		}
		if ir.Tokens == 0 {
			t.Error("zero tokens")
		}
		t.Logf("reply=%q tokens=%d seconds=%.2f", ir.Reply, ir.Tokens, ir.Seconds)
	})

	t.Run("restart_noop", func(t *testing.T) {
		proc2, err := proc.restart("qwen-1.5b", "")
		if err != nil {
			t.Fatalf("restart: %v", err)
		}
		if proc2 != proc {
			t.Error("httpProc.restart should return same instance")
		}
		if proc.model != "qwen-1.5b" {
			t.Errorf("model not updated after restart, got %q", proc.model)
		}
	})
}
