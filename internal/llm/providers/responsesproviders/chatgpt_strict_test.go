package responsesproviders

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/term-llm/term-llm/internal/credentials"
)

func TestChatGPTStreamSendsStrictToolsAndDropsOptionalNulls(t *testing.T) {
	origClient := chatGPTHTTPClient
	defer func() { chatGPTHTTPClient = origClient }()

	var body []byte
	chatGPTHTTPClient = &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		var err error
		body, err = io.ReadAll(req.Body)
		if err != nil {
			t.Fatalf("read request: %v", err)
		}
		args := `{\"code\":\"page.title()\",\"timeout\":null}`
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body: io.NopCloser(strings.NewReader(strings.Join([]string{
				`event: response.output_item.added`,
				`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"run_code"}}`,
				`event: response.output_item.done`,
				`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"run_code","arguments":"` + args + `"}}`,
				`event: response.completed`,
				`data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`,
				`data: [DONE]`,
			}, "\n"))),
		}, nil
	})}

	provider := NewChatGPTProviderWithCreds(&credentials.ChatGPTCredentials{
		AccessToken: "test-token",
		AccountID:   "test-account",
		ExpiresAt:   time.Now().Add(time.Hour).Unix(),
	}, "gpt-6.1-sol-medium")

	stream, err := provider.Stream(context.Background(), Request{
		Messages: []Message{UserText("hello")},
		Tools: []ToolSpec{
			{
				Name: "run_code",
				Schema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"code":    map[string]interface{}{"type": "string"},
						"timeout": map[string]interface{}{"type": "number"},
					},
					"required":             []interface{}{"code"},
					"additionalProperties": false,
				},
			},
			{
				// A free-form map cannot be strict without changing its shape.
				Name: "set_env",
				Schema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"env": map[string]interface{}{"type": "object", "additionalProperties": map[string]interface{}{"type": "string"}},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("stream creation failed: %v", err)
	}
	defer stream.Close()

	var call *ToolCall
	for {
		event, err := stream.Recv()
		if errors.Is(err, io.EOF) || event.Type == EventDone {
			break
		}
		if err != nil {
			t.Fatalf("recv: %v", err)
		}
		if event.Type == EventToolCall {
			call = event.Tool
		}
	}

	var payload struct {
		Tools []struct {
			Name       string         `json:"name"`
			Strict     bool           `json:"strict"`
			Parameters map[string]any `json:"parameters"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	strict := map[string]bool{}
	for _, tool := range payload.Tools {
		strict[tool.Name] = tool.Strict
		if tool.Name == "run_code" && len(tool.Parameters["required"].([]any)) != 2 {
			t.Fatalf("strict run_code should require every property: %#v", tool.Parameters)
		}
	}
	if !strict["run_code"] {
		t.Fatalf("run_code should be sent strict: %s", body)
	}
	if strict["set_env"] {
		t.Fatalf("set_env should fall back to non-strict: %s", body)
	}

	if call == nil {
		t.Fatal("expected a tool call event")
	}
	var args map[string]any
	if err := json.Unmarshal(call.Arguments, &args); err != nil {
		t.Fatalf("decode tool args: %v", err)
	}
	if _, ok := args["timeout"]; ok {
		t.Fatalf("optional null should be dropped before the tool sees it: %s", call.Arguments)
	}
	if args["code"] != "page.title()" {
		t.Fatalf("required argument lost: %s", call.Arguments)
	}
}

func TestBuildChatGPTToolsKeepsExplicitStrictTools(t *testing.T) {
	tools, upgraded := buildChatGPTTools([]ToolSpec{{
		Name:   "explicit",
		Strict: true,
		Schema: map[string]interface{}{"type": "object", "properties": map[string]interface{}{"a": map[string]interface{}{"type": "string"}}},
	}})
	if len(tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(tools))
	}
	if len(upgraded) != 0 {
		t.Fatalf("explicitly strict tools keep their existing lowering, got upgraded %v", upgraded)
	}
}
