package schema

import (
	"bytes"
	"encoding/json"
)

// Automatic OpenAI strict mode for tool schemas that survive the conversion
// without changing what the tool accepts.
//
// Strict mode requires every property to be listed in required and every
// object to set additionalProperties:false. Optional properties are therefore
// made nullable, and DropStrictNulls removes the nulls the model sends for
// them before the arguments reach the tool. Schemas whose meaning would change
// (free-form maps, open objects, $ref, oneOf/allOf, tuples, ...) are left
// non-strict.

// strictUnsafeKeywords cannot be expressed in strict mode without changing the
// schema's meaning, so their presence keeps a tool non-strict.
var strictUnsafeKeywords = []string{
	"$ref", "$defs", "definitions", "allOf", "oneOf", "not",
	"if", "then", "else", "patternProperties", "propertyNames",
	"unevaluatedProperties", "dependentSchemas", "dependentRequired",
	"dependencies", "contains", "minContains", "maxContains", "prefixItems",
	"minProperties", "maxProperties",
}

// StrictToolParameters returns strict-mode parameters for a tool schema, or
// ok=false when the schema cannot be made strict without changing its meaning.
func StrictToolParameters(raw map[string]interface{}) (map[string]interface{}, bool) {
	if len(raw) == 0 || !strictCompatible(raw, true) {
		return nil, false
	}
	parsed, err := ParseToolJSONSchemaMap(raw)
	if err != nil {
		return nil, false
	}
	params := parsed.ToOpenResponsesParameters()
	scrubUnsupportedOpenAISchemaKeywords(params)
	makeOptionalPropertiesNullable(params)
	params = NormalizeSchemaForOpenAI(params)
	if _, ok := params["required"]; !ok {
		params["required"] = []string{}
	}
	return params, true
}

func strictCompatible(node map[string]interface{}, root bool) bool {
	for _, key := range strictUnsafeKeywords {
		if _, ok := node[key]; ok {
			return false
		}
	}
	if root && (!isObjectSchema(node) || node["anyOf"] != nil) {
		return false
	}
	if isObjectSchema(node) && !strictObjectIsClosed(node, root) {
		return false
	}
	return strictChildrenCompatible(node)
}

func isObjectSchema(node map[string]interface{}) bool {
	if node["properties"] != nil {
		return true
	}
	types, _ := normalizedJSONSchemaTypeNames(node["type"])
	for _, t := range types {
		if t == "object" {
			return true
		}
	}
	return false
}

// strictObjectIsClosed reports whether forcing additionalProperties:false keeps
// the object's meaning. Explicitly open objects and free-form maps do not; a
// nested object with no declared properties is treated as free-form too.
func strictObjectIsClosed(node map[string]interface{}, root bool) bool {
	switch ap := node["additionalProperties"].(type) {
	case nil:
		props, _ := node["properties"].(map[string]interface{})
		return root || len(props) > 0
	case bool:
		return !ap
	default:
		return false
	}
}

func strictChildrenCompatible(node map[string]interface{}) bool {
	if props, ok := node["properties"].(map[string]interface{}); ok {
		for _, prop := range props {
			child, ok := prop.(map[string]interface{})
			if !ok || !strictCompatible(child, false) {
				return false
			}
		}
	}
	if items, ok := node["items"]; ok {
		child, ok := items.(map[string]interface{})
		if !ok || !strictCompatible(child, false) {
			return false
		}
	}
	if anyOf, ok := node["anyOf"].([]interface{}); ok {
		for _, branch := range anyOf {
			child, ok := branch.(map[string]interface{})
			if !ok || !strictCompatible(child, false) {
				return false
			}
		}
	}
	return true
}

// makeOptionalPropertiesNullable lets the model send null for properties that
// were optional before strict mode made every property required. It runs
// before NormalizeSchemaForOpenAI, which turns ["T","null"] into anyOf.
func makeOptionalPropertiesNullable(node map[string]interface{}) {
	if props, ok := node["properties"].(map[string]interface{}); ok {
		required := requiredSet(node)
		for key, prop := range props {
			child, ok := prop.(map[string]interface{})
			if !ok {
				continue
			}
			makeOptionalPropertiesNullable(child)
			if !required[key] {
				addNullType(child)
			}
		}
	}
	if items, ok := node["items"].(map[string]interface{}); ok {
		makeOptionalPropertiesNullable(items)
	}
	if anyOf, ok := node["anyOf"].([]interface{}); ok {
		for _, branch := range anyOf {
			if child, ok := branch.(map[string]interface{}); ok {
				makeOptionalPropertiesNullable(child)
			}
		}
	}
}

func addNullType(node map[string]interface{}) {
	if anyOf, ok := node["anyOf"].([]interface{}); ok {
		for _, branch := range anyOf {
			if b, ok := branch.(map[string]interface{}); ok && b["type"] == "null" {
				return
			}
		}
		node["anyOf"] = append(anyOf, map[string]interface{}{"type": "null"})
		return
	}
	types, _ := normalizedJSONSchemaTypeNames(node["type"])
	if len(types) == 0 {
		types = []string{inferJSONSchemaType(node)}
	}
	for _, t := range types {
		if t == "null" {
			node["type"] = toInterfaces(types)
			return
		}
	}
	node["type"] = toInterfaces(append(types, "null"))
}

func toInterfaces(values []string) []interface{} {
	out := make([]interface{}, len(values))
	for i, v := range values {
		out[i] = v
	}
	return out
}

func requiredSet(node map[string]interface{}) map[string]bool {
	set := map[string]bool{}
	switch req := node["required"].(type) {
	case []string:
		for _, key := range req {
			set[key] = true
		}
	case []interface{}:
		for _, key := range req {
			if s, ok := key.(string); ok {
				set[s] = true
			}
		}
	}
	return set
}

// DropStrictNulls removes null values that the model sent for properties the
// original schema marked optional, so tools see the field as omitted. Nulls
// for required properties are kept. Arguments that are not a JSON object, or
// that contain no such nulls, are returned unchanged.
func DropStrictNulls(raw map[string]interface{}, args json.RawMessage) json.RawMessage {
	if len(raw) == 0 || !bytes.Contains(args, []byte("null")) {
		return args
	}
	var value interface{}
	decoder := json.NewDecoder(bytes.NewReader(args))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return args
	}
	if !dropOptionalNulls(raw, value) {
		return args
	}
	out, err := json.Marshal(value)
	if err != nil {
		return args
	}
	return out
}

func dropOptionalNulls(node map[string]interface{}, value interface{}) bool {
	changed := false
	switch v := value.(type) {
	case map[string]interface{}:
		props, _ := node["properties"].(map[string]interface{})
		required := requiredSet(node)
		for key, item := range v {
			if item == nil && !required[key] {
				if _, declared := props[key]; declared {
					delete(v, key)
					changed = true
					continue
				}
			}
			if child, ok := props[key].(map[string]interface{}); ok {
				changed = dropOptionalNulls(child, item) || changed
			}
		}
	case []interface{}:
		items, _ := node["items"].(map[string]interface{})
		for _, item := range v {
			if items != nil {
				changed = dropOptionalNulls(items, item) || changed
			}
		}
	}
	if anyOf, ok := node["anyOf"].([]interface{}); ok {
		for _, branch := range anyOf {
			if child, ok := branch.(map[string]interface{}); ok {
				changed = dropOptionalNulls(child, value) || changed
			}
		}
	}
	return changed
}
