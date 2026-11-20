package responsesproviders

import (
	"github.com/term-llm/term-llm/internal/llm/providers/responses"
	"github.com/term-llm/term-llm/internal/llm/schema"
)

// buildChatGPTTools sends every tool whose schema can be made strict with
// strict:true. Without strict mode, gpt-6.1-sol sometimes streams endless
// whitespace into a JSON argument string (seen with Playwright's
// browser_run_code_unsafe); constrained decoding keeps it inside valid JSON.
// Tools whose schemas would change meaning stay non-strict. The returned map
// holds the original schemas of upgraded tools so optional-field nulls can be
// dropped from their calls.
func buildChatGPTTools(specs []ToolSpec) ([]any, map[string]map[string]interface{}) {
	tools := BuildResponsesTools(specs)
	if len(tools) != len(specs) {
		return tools, nil
	}
	var upgraded map[string]map[string]interface{}
	for i, spec := range specs {
		tool, ok := tools[i].(responses.ResponsesTool)
		if !ok || tool.Strict {
			continue
		}
		params, ok := schema.StrictToolParameters(spec.Schema)
		if !ok {
			continue
		}
		tool.Parameters = params
		tool.Strict = true
		tools[i] = tool
		if upgraded == nil {
			upgraded = make(map[string]map[string]interface{})
		}
		upgraded[spec.Name] = spec.Schema
	}
	return tools, upgraded
}

// strictNullStream drops the nulls strict mode makes the model send for
// optional fields, so tools see those fields as omitted rather than null.
type strictNullStream struct {
	Stream
	schemas map[string]map[string]interface{}
}

func wrapStrictNullStream(stream Stream, schemas map[string]map[string]interface{}) Stream {
	if stream == nil || len(schemas) == 0 {
		return stream
	}
	return &strictNullStream{Stream: stream, schemas: schemas}
}

func (s *strictNullStream) Recv() (Event, error) {
	event, err := s.Stream.Recv()
	if err != nil || event.Type != EventToolCall || event.Tool == nil {
		return event, err
	}
	if raw, ok := s.schemas[event.Tool.Name]; ok {
		call := *event.Tool
		call.Arguments = schema.DropStrictNulls(raw, call.Arguments)
		event.Tool = &call
	}
	return event, nil
}
