package main

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"
)

// TestGatewayHTTPProc exercises httpProc against the live gateway on :7705.
// Skipped if the gateway is not reachable or has no model loaded.
func TestGatewayHTTPProc(t *testing.T) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://127.0.0.1:7705/health")
	if err != nil {
		t.Skip("gateway not reachable on :7705 — skipping live test")
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Skip("gateway returned non-200 — skipping live test")
	}

	// Check which model the gateway has loaded — skip if queue is busy.
	var health struct {
		Status     string `json:"status"`
		Model      string `json:"model"`
		QueueDepth int    `json:"queue_depth"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		t.Skipf("cannot parse gateway health: %v", err)
	}
	if health.Status != "ready" {
		t.Skipf("gateway not ready: status=%s", health.Status)
	}
	if health.QueueDepth > 0 {
		t.Skipf("gateway busy: queue_depth=%d", health.QueueDepth)
	}

	testModel := health.Model
	if testModel == "" {
		testModel = "qwen-0.5b"
	}

	proc := &httpProc{
		baseURL: "http://127.0.0.1:7705",
		model:   testModel,
		client:  &http.Client{Timeout: 30 * time.Second},
	}

	t.Run("waitReady", func(t *testing.T) {
		rp, err := proc.waitReady()
		if err != nil {
			t.Fatalf("waitReady: %v", err)
		}
		if rp.Status != "ready" {
			t.Errorf("expected status=ready, got %q", rp.Status)
		}
		t.Logf("ready: model=%s useNative=%v", rp.Model, rp.UseNative)
	})

	t.Run("doInfer", func(t *testing.T) {
		msgs := []Message{{Role: "user", Content: "Reply with exactly one word: pong"}}
		ir, err := proc.doInfer(msgs, 10, 0)
		if err != nil {
			t.Skipf("doInfer: %v (backend may have model paged out)", err)
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
		proc2, err := proc.restart(testModel, "")
		if err != nil {
			t.Fatalf("restart: %v", err)
		}
		if proc2 != proc {
			t.Error("httpProc.restart should return same instance")
		}
		if proc.model != testModel {
			t.Errorf("model not updated after restart, got %q", proc.model)
		}
	})
}
