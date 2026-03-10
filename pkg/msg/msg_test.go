package msg

import "testing"

func TestStripMarkupNestedJSON(t *testing.T) {
	input := `Some text TOOL_CALL: {"name":"write_file","arguments":{"path":"x.json","content":"{\"key\":\"val\"}"}} more text`
	got := StripMarkup(input)
	want := "Some text  more text"
	if got != want {
		t.Errorf("StripMarkup nested JSON:\n got: %q\nwant: %q", got, want)
	}
}

func TestStripMarkupSimple(t *testing.T) {
	input := `Hello TOOL_CALL: {"name":"list_files","arguments":{"pattern":"*.go"}} world`
	got := StripMarkup(input)
	want := "Hello  world"
	if got != want {
		t.Errorf("StripMarkup simple:\n got: %q\nwant: %q", got, want)
	}
}

func TestStripMarkupNative(t *testing.T) {
	input := `Hello <tool_call>{"name":"test"}</tool_call> world`
	got := StripMarkup(input)
	want := "Hello  world"
	if got != want {
		t.Errorf("StripMarkup native:\n got: %q\nwant: %q", got, want)
	}
}

func TestStripMarkupMultiple(t *testing.T) {
	input := `TOOL_CALL: {"name":"a","arguments":{}} text TOOL_CALL: {"name":"b","arguments":{}}`
	got := StripMarkup(input)
	want := "text"
	if got != want {
		t.Errorf("StripMarkup multiple:\n got: %q\nwant: %q", got, want)
	}
}

func TestStripMarkupBracesInArgs(t *testing.T) {
	input := `TOOL_CALL: {"name":"shell","arguments":{"cmd":"echo {hello} world"}}`
	got := StripMarkup(input)
	want := ""
	if got != want {
		t.Errorf("StripMarkup braces in args:\n got: %q\nwant: %q", got, want)
	}
}
