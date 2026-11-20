package schema

import (
	"encoding/json"
	"reflect"
	"sort"
	"testing"
)

func decodeSchema(t *testing.T, raw string) map[string]interface{} {
	t.Helper()
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		t.Fatalf("decode schema: %v", err)
	}
	return m
}

func sortedRequired(t *testing.T, node map[string]interface{}) []string {
	t.Helper()
	var out []string
	switch req := node["required"].(type) {
	case []string:
		out = append(out, req...)
	case []interface{}:
		for _, v := range req {
			out = append(out, v.(string))
		}
	default:
		t.Fatalf("required missing or wrong type: %#v", node["required"])
	}
	sort.Strings(out)
	return out
}

func hasNullBranch(node map[string]interface{}) bool {
	anyOf, _ := node["anyOf"].([]interface{})
	for _, branch := range anyOf {
		if b, ok := branch.(map[string]interface{}); ok && b["type"] == "null" {
			return true
		}
	}
	return false
}

func TestStrictToolParametersMakesOptionalFieldsNullable(t *testing.T) {
	raw := decodeSchema(t, `{
		"type": "object",
		"properties": {
			"code": {"type": "string", "description": "JS to run"},
			"timeout": {"type": "number"},
			"mode": {"type": "string", "enum": ["fast", "slow"]},
			"target": {
				"type": "object",
				"properties": {"ref": {"type": "string"}, "frame": {"type": "integer"}},
				"required": ["ref"],
				"additionalProperties": false
			}
		},
		"required": ["code"],
		"additionalProperties": false
	}`)

	params, ok := StrictToolParameters(raw)
	if !ok {
		t.Fatal("expected schema to be strict-compatible")
	}
	if params["additionalProperties"] != false {
		t.Fatalf("root additionalProperties = %#v, want false", params["additionalProperties"])
	}
	if got, want := sortedRequired(t, params), []string{"code", "mode", "target", "timeout"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("required = %v, want %v", got, want)
	}

	props := params["properties"].(map[string]interface{})
	code := props["code"].(map[string]interface{})
	if code["type"] != "string" || hasNullBranch(code) {
		t.Fatalf("required field should stay non-nullable: %#v", code)
	}
	for _, name := range []string{"timeout", "mode", "target"} {
		if !hasNullBranch(props[name].(map[string]interface{})) {
			t.Fatalf("optional field %q should accept null: %#v", name, props[name])
		}
	}

	// The enum belongs to the string branch only, so null stays a valid value.
	mode := props["mode"].(map[string]interface{})
	for _, branch := range mode["anyOf"].([]interface{}) {
		b := branch.(map[string]interface{})
		if b["type"] == "string" && b["enum"] == nil {
			t.Fatalf("enum dropped from string branch: %#v", mode)
		}
		if b["type"] == "null" && b["enum"] != nil {
			t.Fatalf("enum copied onto null branch: %#v", mode)
		}
	}

	// Nested objects get the same treatment.
	var target map[string]interface{}
	for _, branch := range props["target"].(map[string]interface{})["anyOf"].([]interface{}) {
		if b := branch.(map[string]interface{}); b["type"] == "object" {
			target = b
		}
	}
	if target == nil || target["additionalProperties"] != false {
		t.Fatalf("nested object not closed: %#v", props["target"])
	}
	if got, want := sortedRequired(t, target), []string{"frame", "ref"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("nested required = %v, want %v", got, want)
	}
	if !hasNullBranch(target["properties"].(map[string]interface{})["frame"].(map[string]interface{})) {
		t.Fatalf("nested optional field should accept null: %#v", target)
	}

	// The caller's schema must not be mutated.
	if _, ok := raw["properties"].(map[string]interface{})["timeout"].(map[string]interface{})["anyOf"]; ok {
		t.Fatal("input schema was mutated")
	}
}

func TestStrictToolParametersNoArgTool(t *testing.T) {
	params, ok := StrictToolParameters(decodeSchema(t, `{"type": "object", "properties": {}}`))
	if !ok {
		t.Fatal("expected a no-argument tool to be strict-compatible")
	}
	if params["additionalProperties"] != false {
		t.Fatalf("additionalProperties = %#v, want false", params["additionalProperties"])
	}
	if req, ok := params["required"].([]string); !ok || len(req) != 0 {
		t.Fatalf("required = %#v, want empty list", params["required"])
	}
}

func TestStrictToolParametersFallsBackForUnsupportedSchemas(t *testing.T) {
	cases := map[string]string{
		"empty schema":       `{}`,
		"non-object root":    `{"type": "string"}`,
		"free-form map":      `{"type": "object", "properties": {"env": {"type": "object", "additionalProperties": {"type": "string"}}}}`,
		"open root":          `{"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": true}`,
		"untyped nested obj": `{"type": "object", "properties": {"opts": {"type": "object"}}}`,
		"ref":                `{"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}, "$defs": {"A": {"type": "string"}}}`,
		"oneOf":              `{"type": "object", "properties": {"a": {"oneOf": [{"type": "string"}, {"type": "number"}]}}}`,
		"allOf":              `{"type": "object", "properties": {"a": {"allOf": [{"type": "string"}]}}}`,
		"patternProperties":  `{"type": "object", "patternProperties": {"^x-": {"type": "string"}}}`,
		"tuple items":        `{"type": "object", "properties": {"pt": {"type": "array", "items": [{"type": "number"}, {"type": "number"}]}}}`,
		"boolean property":   `{"type": "object", "properties": {"anything": true}}`,
		"nested in array":    `{"type": "object", "properties": {"rows": {"type": "array", "items": {"type": "object", "additionalProperties": {"type": "string"}}}}}`,
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			if params, ok := StrictToolParameters(decodeSchema(t, raw)); ok {
				t.Fatalf("expected non-strict fallback, got %#v", params)
			}
		})
	}
}

func TestDropStrictNulls(t *testing.T) {
	raw := decodeSchema(t, `{
		"type": "object",
		"properties": {
			"code": {"type": ["string", "null"]},
			"timeout": {"type": "number"},
			"target": {"type": "object", "properties": {"ref": {"type": "string"}, "frame": {"type": "integer"}}, "required": ["ref"]},
			"steps": {"type": "array", "items": {"type": "object", "properties": {"wait": {"type": "number"}}}}
		},
		"required": ["code"]
	}`)

	got := DropStrictNulls(raw, json.RawMessage(`{"code":null,"timeout":null,"target":{"ref":"e1","frame":null},"steps":[{"wait":null},{"wait":2}],"extra":null}`))
	var decoded map[string]interface{}
	if err := json.Unmarshal(got, &decoded); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	want := map[string]interface{}{
		"code":   nil, // required, so an explicit null is the model's real answer
		"target": map[string]interface{}{"ref": "e1"},
		"steps":  []interface{}{map[string]interface{}{}, map[string]interface{}{"wait": float64(2)}},
		"extra":  nil, // undeclared keys are not ours to touch
	}
	if !reflect.DeepEqual(decoded, want) {
		t.Fatalf("DropStrictNulls = %s, want %#v", got, want)
	}

	unchanged := json.RawMessage(`{"code": "1", "timeout": 3}`)
	if out := DropStrictNulls(raw, unchanged); string(out) != string(unchanged) {
		t.Fatalf("args without nulls should pass through untouched, got %s", out)
	}
	if out := DropStrictNulls(raw, json.RawMessage(`not json null`)); string(out) != `not json null` {
		t.Fatalf("invalid JSON should pass through untouched, got %s", out)
	}
}
