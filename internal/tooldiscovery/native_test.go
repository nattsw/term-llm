package tooldiscovery

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/term-llm/term-llm/internal/config"
	"github.com/term-llm/term-llm/internal/credentials"
	_ "github.com/term-llm/term-llm/internal/llm"
	"github.com/term-llm/term-llm/internal/llm/core"
	"github.com/term-llm/term-llm/internal/llm/engine"
	"github.com/term-llm/term-llm/internal/llm/providers/responsesproviders"
)

type nativeScriptTurn struct {
	events []core.Event
	err    error
}

type nativeScriptProvider struct {
	mu        sync.Mutex
	supported bool
	turns     []nativeScriptTurn
	requests  []core.Request
	resets    int
}

func (p *nativeScriptProvider) Name() string       { return "native-script" }
func (p *nativeScriptProvider) Credential() string { return "mock" }
func (p *nativeScriptProvider) Capabilities() core.Capabilities {
	return core.Capabilities{ToolCalls: true}
}
func (p *nativeScriptProvider) NativeToolDiscoverySupport(string) core.NativeToolDiscoverySupport {
	if p.supported {
		return core.NativeToolDiscoverySupport{Supported: true, Name: "test-native", Reason: "test capability"}
	}
	return core.NativeToolDiscoverySupport{Reason: "test provider does not support native discovery"}
}
func (p *nativeScriptProvider) ResetConversation() {
	p.mu.Lock()
	p.resets++
	p.mu.Unlock()
}
func (p *nativeScriptProvider) Stream(_ context.Context, req core.Request) (core.Stream, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.requests = append(p.requests, req)
	if len(p.turns) == 0 {
		return nil, errors.New("native script exhausted")
	}
	turn := p.turns[0]
	p.turns = p.turns[1:]
	if turn.err != nil {
		return nil, turn.err
	}
	return &nativeScriptStream{events: append([]core.Event(nil), turn.events...)}, nil
}

type nativeScriptStream struct {
	events []core.Event
}

func (s *nativeScriptStream) Recv() (core.Event, error) {
	if len(s.events) == 0 {
		return core.Event{}, io.EOF
	}
	event := s.events[0]
	s.events = s.events[1:]
	return event, nil
}
func (s *nativeScriptStream) Close() error { return nil }

func TestNativeStrategyOrchestratesDiscoveryWithoutTopLevelSchemaMutation(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true, turns: []nativeScriptTurn{
		{events: []core.Event{{Type: core.EventDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-1", Arguments: json.RawMessage(`{"tool_names":["special_action"]}`)}}}},
		{events: []core.Event{{Type: core.EventToolCall, Tool: &core.ToolCall{ID: "external-1", Namespace: "federation", ChildName: "special_action", Arguments: json.RawMessage(`{"project_id":"P-42"}`)}}}},
		{events: []core.Event{{Type: core.EventTextDelta, Text: "done"}}},
	}}
	eng := engine.NewEngine(provider, nil)
	if _, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "auto", Strategy: "auto", Threshold: 24}, manager, eng); err != nil {
		t.Fatal(err)
	}
	stream, err := eng.Stream(context.Background(), core.Request{
		EnableToolDiscovery: true,
		SessionID:           "native",
		Model:               "native-model",
		Messages:            []core.Message{core.UserText("perform the special action")},
		MaxTurns:            3,
	})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)

	if len(provider.requests) != 3 {
		t.Fatalf("provider requests = %d, want 3", len(provider.requests))
	}
	if provider.requests[0].NativeToolDiscovery == nil {
		t.Fatal("initial request omitted native discovery control")
	}
	for i, request := range provider.requests {
		if len(request.Tools) != 0 {
			t.Fatalf("request %d top-level tools = %v, want stable empty ordinary surface", i, toolNames(request.Tools))
		}
	}
	if provider.requests[1].NativeToolDiscovery != nil {
		t.Fatal("native discovery remained enabled after final-turn cutoff")
	}
	var callSeen, outputSeen bool
	for _, message := range provider.requests[1].Messages {
		for _, part := range message.Parts {
			if part.Type == core.PartDiscoveryCall && part.DiscoveryCall != nil {
				callSeen = true
			}
			if part.Type == core.PartDiscoveryOutput && part.DiscoveryOutput != nil {
				outputSeen = true
				if len(part.DiscoveryOutput.Tools) != 1 || part.DiscoveryOutput.Tools[0].Spec.Name != "federation__special_action" {
					t.Fatalf("discovery output = %#v", part.DiscoveryOutput)
				}
			}
		}
	}
	if !callSeen || !outputSeen {
		t.Fatalf("native replay parts call=%v output=%v", callSeen, outputSeen)
	}
	var routedCall *core.ToolCall
	for _, message := range provider.requests[2].Messages {
		for _, part := range message.Parts {
			if part.Type == core.PartToolCall && part.ToolCall != nil && part.ToolCall.ID == "external-1" {
				routedCall = part.ToolCall
			}
		}
	}
	if routedCall == nil || routedCall.Name != "federation__special_action" || routedCall.Namespace != "federation" || routedCall.ChildName != "special_action" {
		t.Fatalf("persisted native route = %#v", routedCall)
	}
	diagnostics, _ := eng.ToolDiscoveryDiagnostics("native")
	if diagnostics.Strategy != string(StrategyNative) || diagnostics.DynamicActive != 1 || diagnostics.FallbackCount != 0 {
		t.Fatalf("native diagnostics = %+v", diagnostics)
	}
}

func TestNativeHistoryDoesNotRestoreEvictedTools(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true}
	eng := engine.NewEngine(provider, nil)
	planner, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native", MaxActiveTools: 1}, manager, eng)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := manager.CatalogueSnapshot()
	if snapshot == nil || len(snapshot.Tools) < 2 {
		t.Fatal("expected at least two catalogue tools")
	}
	first, second := snapshot.Tools[0], snapshot.Tools[1]
	req := core.Request{SessionID: "native-eviction", Messages: []core.Message{{Role: core.RoleAssistant, Parts: []core.Part{
		{Type: core.PartDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-old"}},
		{Type: core.PartDiscoveryOutput, DiscoveryOutput: &core.ToolDiscoveryOutput{CallID: "search-old", Tools: []core.DiscoveredTool{
			{Spec: first.ToolSpec(), SchemaHash: first.SchemaHash},
			{Spec: second.ToolSpec(), SchemaHash: second.SchemaHash},
		}}},
	}}}}
	key := planner.stateKey(req.SessionID, "")
	planner.mu.Lock()
	planner.stateLocked(key).evicted[first.Name] = true
	planner.mu.Unlock()

	originalOutput := req.Messages[0].Parts[1].DiscoveryOutput
	planner.restoreNativeHistory(&req, key)
	output := req.Messages[0].Parts[1].DiscoveryOutput
	if output == nil || len(output.Tools) != 1 || output.Tools[0].Spec.Name != second.Name {
		t.Fatalf("restored discovery output = %+v, want only %q", output, second.Name)
	}
	if len(originalOutput.Tools) != 2 {
		t.Fatalf("caller-owned discovery output was mutated: %+v", originalOutput)
	}
	got := toolNames(planner.ActiveToolSpecs(req.SessionID))
	if len(got) != 1 || got[0] != second.Name {
		t.Fatalf("restored active tools = %v, want [%s]", got, second.Name)
	}

	emptyOutput := &core.ToolDiscoveryOutput{CallID: "search-empty", Tools: []core.DiscoveredTool{
		{Spec: first.ToolSpec(), SchemaHash: first.SchemaHash},
		{Spec: second.ToolSpec(), SchemaHash: second.SchemaHash},
	}}
	emptyReq := core.Request{SessionID: "native-all-evicted", Messages: []core.Message{{Role: core.RoleAssistant, Parts: []core.Part{
		{Type: core.PartDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-empty"}},
		{Type: core.PartDiscoveryOutput, DiscoveryOutput: emptyOutput},
	}}}}
	emptyKey := planner.stateKey(emptyReq.SessionID, "")
	planner.mu.Lock()
	emptyState := planner.stateLocked(emptyKey)
	emptyState.evicted[first.Name] = true
	emptyState.evicted[second.Name] = true
	planner.mu.Unlock()
	planner.restoreNativeHistory(&emptyReq, emptyKey)
	if len(emptyReq.Messages[0].Parts) != 0 {
		t.Fatalf("empty discovery call/output pair was retained: %+v", emptyReq.Messages[0].Parts)
	}
	if len(emptyOutput.Tools) != 2 {
		t.Fatalf("caller-owned empty discovery output was mutated: %+v", emptyOutput)
	}
}

func TestNativeNamespaceCallRejectsUnselectedSibling(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true, turns: []nativeScriptTurn{
		{events: []core.Event{{Type: core.EventDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-1", Arguments: json.RawMessage(`{"tool_names":["special_action"]}`)}}}},
		{events: []core.Event{{Type: core.EventToolCall, Tool: &core.ToolCall{ID: "sibling-1", Namespace: "federation", ChildName: "realistic_operation_00", Arguments: json.RawMessage(`{}`)}}}},
	}}
	eng := engine.NewEngine(provider, nil)
	if _, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native"}, manager, eng); err != nil {
		t.Fatal(err)
	}
	stream, err := eng.Stream(context.Background(), core.Request{
		EnableToolDiscovery: true,
		SessionID:           "unselected-sibling",
		Model:               "native-model",
		Messages:            []core.Message{core.UserText("act")},
		MaxTurns:            3,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var gotErr error
	for {
		event, recvErr := stream.Recv()
		if recvErr != nil {
			if !errors.Is(recvErr, io.EOF) {
				gotErr = recvErr
			}
			break
		}
		if event.Type == core.EventError {
			gotErr = event.Err
			break
		}
	}
	if gotErr == nil || !strings.Contains(gotErr.Error(), "not loaded and authorised") {
		t.Fatalf("unselected sibling error = %v", gotErr)
	}
	if len(provider.requests) != 2 {
		t.Fatalf("provider continued after rejected sibling: %d requests", len(provider.requests))
	}
}

func TestNativeAutoFallbackOccursOnceBeforeSideEffects(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true, turns: []nativeScriptTurn{
		{err: errors.New("native tool_search rejected")},
		{events: []core.Event{{Type: core.EventToolCall, Tool: &core.ToolCall{ID: "portable-search", Name: ToolSearchName, Arguments: json.RawMessage(`{"tool_names":["special_action"]}`)}}}},
		{events: []core.Event{{Type: core.EventToolCall, Tool: &core.ToolCall{ID: "external", Name: "federation__special_action", Arguments: json.RawMessage(`{"project_id":"P-1"}`)}}}},
		{events: []core.Event{{Type: core.EventTextDelta, Text: "done"}}},
	}}
	eng := engine.NewEngine(provider, nil)
	if _, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "auto"}, manager, eng); err != nil {
		t.Fatal(err)
	}
	stream, err := eng.Stream(context.Background(), core.Request{EnableToolDiscovery: true, SessionID: "fallback", Model: "native-model", Messages: []core.Message{core.UserText("act")}, MaxTurns: 4})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)
	if len(provider.requests) != 4 || provider.requests[0].NativeToolDiscovery == nil || !hasTool(provider.requests[1].Tools, ToolSearchName) {
		t.Fatalf("fallback requests = %#v", provider.requests)
	}
	diagnostics, _ := eng.ToolDiscoveryDiagnostics("fallback")
	if diagnostics.Strategy != string(StrategyPortable) || diagnostics.FallbackCount != 1 || !strings.Contains(diagnostics.FallbackReason, "rejected") {
		t.Fatalf("fallback diagnostics = %+v", diagnostics)
	}
	if provider.resets != 1 {
		t.Fatalf("provider resets = %d, want 1", provider.resets)
	}
}

func TestAutoStrategyUsesPortableWithoutProvenNativeCapability(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: false}
	eng := engine.NewEngine(provider, nil)
	planner, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "auto"}, manager, eng)
	if err != nil {
		t.Fatal(err)
	}
	req := core.Request{SessionID: "qwen-like", Model: "qwen"}
	if _, err := planner.BeginRun(context.Background(), provider, &req, "qwen-run"); err != nil {
		t.Fatal(err)
	}
	defer planner.EndRun("qwen-run")
	if req.NativeToolDiscovery != nil || !hasTool(req.Tools, ToolSearchName) {
		t.Fatalf("auto fallback surface native=%v tools=%v", req.NativeToolDiscovery != nil, toolNames(req.Tools))
	}
	diagnostics := planner.Diagnostics("qwen-like")
	if diagnostics.Strategy != string(StrategyPortable) || !strings.Contains(diagnostics.StrategyReason, "not proven") {
		t.Fatalf("portable diagnostics = %+v", diagnostics)
	}
}

func TestChatGPTLunaAutoStrategyResolvesNative(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := responsesproviders.NewChatGPTProviderWithCredsAndOptions(
		&credentials.ChatGPTCredentials{AccessToken: "test", ExpiresAt: time.Now().Add(time.Hour).Unix()},
		"gpt-5.6-luna-medium",
		responsesproviders.ChatGPTProviderOptions{UseWebSocket: true},
	)
	eng := engine.NewEngine(provider, nil)
	planner, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "auto", Strategy: "auto", Threshold: 24}, manager, eng)
	if err != nil {
		t.Fatal(err)
	}
	req := core.Request{SessionID: "luna", Model: "gpt-5.6-luna-medium"}
	if _, err := planner.BeginRun(context.Background(), provider, &req, "luna-run"); err != nil {
		t.Fatal(err)
	}
	defer planner.EndRun("luna-run")
	if req.NativeToolDiscovery == nil || hasTool(req.Tools, ToolSearchName) {
		t.Fatalf("Luna auto surface native=%v tools=%v", req.NativeToolDiscovery != nil, toolNames(req.Tools))
	}
	if got := planner.Diagnostics("luna").Strategy; got != string(StrategyNative) {
		t.Fatalf("Luna strategy = %q, want native", got)
	}
}

func TestNativeReplayStaleSchemaIsInvalidatedAndReset(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true}
	eng := engine.NewEngine(provider, nil)
	planner, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native"}, manager, eng)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := manager.CatalogueSnapshot()
	var spec core.ToolSpec
	for _, tool := range snapshot.Tools {
		if tool.Name == "federation__special_action" {
			spec = tool.ToolSpec()
			break
		}
	}
	req := core.Request{SessionID: "stale", Model: "native-model", Messages: []core.Message{{Role: core.RoleAssistant, Parts: []core.Part{
		{Type: core.PartDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-old", Arguments: json.RawMessage(`{"query":"special"}`)}},
		{Type: core.PartDiscoveryOutput, DiscoveryOutput: &core.ToolDiscoveryOutput{CallID: "search-old", CatalogueGen: snapshot.Generation - 1, Tools: []core.DiscoveredTool{{Spec: spec, SchemaHash: "stale-hash"}}}},
	}}}}
	resetReason, err := planner.BeginRun(context.Background(), provider, &req, "stale-run")
	if err != nil {
		t.Fatal(err)
	}
	defer planner.EndRun("stale-run")
	if !strings.Contains(resetReason, "schema changed") {
		t.Fatalf("reset reason = %q", resetReason)
	}
	for _, message := range req.Messages {
		for _, part := range message.Parts {
			if part.Type == core.PartDiscoveryCall || part.Type == core.PartDiscoveryOutput {
				t.Fatalf("stale discovery replay survived: %#v", part)
			}
		}
	}
}

func TestNativeReplayOldGenerationReResolvesCurrentCatalogue(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true}
	eng := engine.NewEngine(provider, nil)
	planner, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native"}, manager, eng)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := manager.CatalogueSnapshot()
	var selected core.DiscoveredTool
	for _, tool := range snapshot.Tools {
		if tool.Name == "federation__special_action" {
			selected = core.DiscoveredTool{Spec: tool.ToolSpec(), SchemaHash: tool.SchemaHash}
			break
		}
	}
	output := &core.ToolDiscoveryOutput{CallID: "search-old", CatalogueHash: "old", CatalogueGen: 0, Tools: []core.DiscoveredTool{selected}}
	req := core.Request{SessionID: "old-generation", Model: "native-model", Messages: []core.Message{{Role: core.RoleAssistant, Parts: []core.Part{
		{Type: core.PartDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-old", Arguments: json.RawMessage(`{"query":"special"}`)}},
		{Type: core.PartDiscoveryOutput, DiscoveryOutput: output},
	}}}}
	resetReason, err := planner.BeginRun(context.Background(), provider, &req, "old-generation-run")
	if err != nil {
		t.Fatal(err)
	}
	defer planner.EndRun("old-generation-run")
	if resetReason != "" {
		t.Fatalf("unchanged schema forced reset: %q", resetReason)
	}
	resolvedOutput := req.Messages[0].Parts[1].DiscoveryOutput
	if resolvedOutput == nil || resolvedOutput.CatalogueGen != snapshot.Generation || resolvedOutput.CatalogueHash != snapshot.Hash || resolvedOutput.Tools[0].Spec.Name != "federation__special_action" {
		t.Fatalf("re-resolved output = %#v", resolvedOutput)
	}
	if output.CatalogueGen != 0 || output.CatalogueHash != "old" {
		t.Fatalf("caller-owned output was mutated = %#v", output)
	}
}

func TestNativeSearchCannotLoadDeniedTool(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true}
	eng := engine.NewEngine(provider, nil)
	eng.SetAllowedToolsFilter([]string{})
	planner, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native"}, manager, eng)
	if err != nil {
		t.Fatal(err)
	}
	req := core.Request{SessionID: "denied", Model: "native-model"}
	if _, err := planner.BeginRun(context.Background(), provider, &req, "denied-run"); err != nil {
		t.Fatal(err)
	}
	defer planner.EndRun("denied-run")
	if _, err := planner.PrepareTurn(context.Background(), provider, &req, "denied-run", 0, 3); err != nil {
		t.Fatal(err)
	}
	output, err := planner.ResolveNativeToolDiscovery(context.Background(), "denied-run", core.ToolDiscoveryCall{ID: "search", Arguments: json.RawMessage(`{"tool_names":["special_action"]}`)})
	if err != nil {
		t.Fatalf("denied search failed the run: %v", err)
	}
	if output.CallID != "search" || len(output.Tools) != 0 {
		t.Fatalf("denied search output = %#v, want an empty result for call search", output)
	}
}

func TestNativeFallbackRejectedAfterCommit(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true}
	eng := engine.NewEngine(provider, nil)
	planner, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "auto"}, manager, eng)
	if err != nil {
		t.Fatal(err)
	}
	req := core.Request{SessionID: "committed", Model: "native-model"}
	if _, err := planner.BeginRun(context.Background(), provider, &req, "committed-run"); err != nil {
		t.Fatal(err)
	}
	defer planner.EndRun("committed-run")
	if fallback, _ := planner.FallbackNativeToolDiscovery("committed-run", errors.New("late failure"), true); fallback {
		t.Fatal("native fallback succeeded after a committed side effect")
	}
}

func TestForcedNativeUnsupportedProviderFailsClearly(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: false, turns: []nativeScriptTurn{{events: []core.Event{{Type: core.EventTextDelta, Text: "unexpected"}}}}}
	eng := engine.NewEngine(provider, nil)
	if _, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native"}, manager, eng); err != nil {
		t.Fatal(err)
	}
	stream, err := eng.Stream(context.Background(), core.Request{EnableToolDiscovery: true, Model: "unsupported", Messages: []core.Message{core.UserText("act")}, MaxTurns: 3})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var gotErr error
	for {
		event, recvErr := stream.Recv()
		if recvErr != nil {
			if !errors.Is(recvErr, io.EOF) {
				gotErr = recvErr
			}
			break
		}
		if event.Type == core.EventError && event.Err != nil {
			gotErr = event.Err
			break
		}
	}
	if gotErr == nil || !strings.Contains(gotErr.Error(), "strategy native is unsupported") {
		t.Fatalf("error = %v", gotErr)
	}
	if len(provider.requests) != 0 {
		t.Fatalf("unsupported forced native reached provider: %d requests", len(provider.requests))
	}
}

// A forced tool_search is satisfied by a native discovery call. The engine must
// keep that discovery in history and move on, not treat the discovery-only turn
// as an ignored force and re-issue the search with the discovery discarded.
func TestNativeForcedToolSearchKeepsDiscoveryAndDoesNotRepeat(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true, turns: []nativeScriptTurn{
		{events: []core.Event{{Type: core.EventDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-1", Arguments: json.RawMessage(`{"tool_names":["special_action"]}`)}}}},
		{events: []core.Event{{Type: core.EventToolCall, Tool: &core.ToolCall{ID: "external-1", Namespace: "federation", ChildName: "special_action", Arguments: json.RawMessage(`{"project_id":"P-42"}`)}}}},
		{events: []core.Event{{Type: core.EventTextDelta, Text: "done"}}},
	}}
	eng := engine.NewEngine(provider, nil)
	if _, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native"}, manager, eng); err != nil {
		t.Fatal(err)
	}
	stream, err := eng.Stream(context.Background(), core.Request{
		EnableToolDiscovery: true,
		SessionID:           "forced-native",
		Model:               "native-model",
		Messages:            []core.Message{core.UserText("perform the special action")},
		Tools:               []core.ToolSpec{(&SearchTool{}).Spec()},
		ToolChoice:          core.ToolChoice{Mode: core.ToolChoiceName, Name: ToolSearchName},
		MaxTurns:            4,
	})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)

	if len(provider.requests) != 3 {
		t.Fatalf("provider requests = %d, want 3 (search, tool call, answer)", len(provider.requests))
	}
	second := provider.requests[1]
	if second.ToolChoice.Mode == core.ToolChoiceName && second.ToolChoice.Name == ToolSearchName {
		t.Fatalf("second request re-forced tool_search: %+v", second.ToolChoice)
	}
	var discoverySeen bool
	for _, message := range second.Messages {
		for _, part := range message.Parts {
			if part.Type == core.PartDiscoveryCall && part.DiscoveryCall != nil && part.DiscoveryCall.ID == "search-1" {
				discoverySeen = true
			}
		}
	}
	if !discoverySeen {
		t.Fatal("second request dropped the forced turn's discovery call from history")
	}
}

// Native discovery presents tools as namespace children, so models may request
// them by "namespace.child". Every exact spelling must resolve to the same tool.
func TestNativeDiscoveryResolvesQualifiedToolNames(t *testing.T) {
	for _, requested := range []string{"special_action", "federation__special_action", "federation.special_action", "federation/special_action", "Federation.Special_Action"} {
		t.Run(requested, func(t *testing.T) {
			manager := startDiscoveryTestManager(t)
			defer manager.StopAll()
			args, _ := json.Marshal(map[string]any{"tool_names": []string{requested}})
			provider := &nativeScriptProvider{supported: true, turns: []nativeScriptTurn{
				{events: []core.Event{{Type: core.EventDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-1", Arguments: args}}}},
				{events: []core.Event{{Type: core.EventTextDelta, Text: "loaded"}}},
			}}
			eng := engine.NewEngine(provider, nil)
			if _, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native"}, manager, eng); err != nil {
				t.Fatal(err)
			}
			stream, err := eng.Stream(context.Background(), core.Request{
				EnableToolDiscovery: true,
				SessionID:           "qualified",
				Model:               "native-model",
				Messages:            []core.Message{core.UserText("load it")},
				MaxTurns:            4,
			})
			if err != nil {
				t.Fatal(err)
			}
			drainStream(t, stream)
			if len(provider.requests) != 2 {
				t.Fatalf("provider requests = %d, want 2", len(provider.requests))
			}
			var loaded []string
			for _, message := range provider.requests[1].Messages {
				for _, part := range message.Parts {
					if part.Type == core.PartDiscoveryOutput && part.DiscoveryOutput != nil {
						for _, tool := range part.DiscoveryOutput.Tools {
							loaded = append(loaded, tool.Spec.Name)
						}
					}
				}
			}
			if len(loaded) != 1 || loaded[0] != "federation__special_action" {
				t.Fatalf("discovery output tools = %v, want [federation__special_action]", loaded)
			}
		})
	}
}

// With a small turn budget the turn after a forced native discovery is already
// past the discovery cutoff. The satisfied force must not reach PrepareTurn,
// which rejects forced tool_search after the cutoff.
func TestNativeForcedToolSearchWithinSmallTurnBudget(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true, turns: []nativeScriptTurn{
		{events: []core.Event{{Type: core.EventDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-1", Arguments: json.RawMessage(`{"tool_names":["special_action"]}`)}}}},
		{events: []core.Event{{Type: core.EventToolCall, Tool: &core.ToolCall{ID: "external-1", Namespace: "federation", ChildName: "special_action", Arguments: json.RawMessage(`{"project_id":"P-42"}`)}}}},
		{events: []core.Event{{Type: core.EventTextDelta, Text: "done"}}},
	}}
	eng := engine.NewEngine(provider, nil)
	if _, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native"}, manager, eng); err != nil {
		t.Fatal(err)
	}
	stream, err := eng.Stream(context.Background(), core.Request{
		EnableToolDiscovery: true,
		SessionID:           "forced-small",
		Model:               "native-model",
		Messages:            []core.Message{core.UserText("perform the special action")},
		Tools:               []core.ToolSpec{(&SearchTool{}).Spec()},
		ToolChoice:          core.ToolChoice{Mode: core.ToolChoiceName, Name: ToolSearchName},
		MaxTurns:            3,
	})
	if err != nil {
		t.Fatal(err)
	}
	drainStream(t, stream)
	if len(provider.requests) != 3 {
		t.Fatalf("provider requests = %d, want 3", len(provider.requests))
	}
}

// drainNativeStream reads a run to the end and fails on any engine error,
// including errors delivered as events.
func drainNativeStream(t *testing.T, stream core.Stream) {
	t.Helper()
	defer stream.Close()
	for {
		event, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return
		}
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if event.Type == core.EventError && event.Err != nil {
			t.Fatalf("stream error event: %v", event.Err)
		}
	}
}

func nativeDiscoveryOutputs(messages []core.Message) []*core.ToolDiscoveryOutput {
	var outputs []*core.ToolDiscoveryOutput
	for _, message := range messages {
		for _, part := range message.Parts {
			if part.Type == core.PartDiscoveryOutput && part.DiscoveryOutput != nil {
				outputs = append(outputs, part.DiscoveryOutput)
			}
		}
	}
	return outputs
}

// Models often write a short commentary line ("I will look up...") in the same
// response as a native tool_search. The discovered tools only reach the model
// on the next turn, so the text must not end the run.
func TestNativeDiscoveryWithCommentaryTextContinuesRun(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true, turns: []nativeScriptTurn{
		{events: []core.Event{
			{Type: core.EventTextDelta, Text: "I will look up the special action tool."},
			{Type: core.EventDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-1", Arguments: json.RawMessage(`{"tool_names":["special_action"]}`)}},
		}},
		{events: []core.Event{{Type: core.EventToolCall, Tool: &core.ToolCall{ID: "external-1", Namespace: "federation", ChildName: "special_action", Arguments: json.RawMessage(`{"project_id":"P-42"}`)}}}},
		{events: []core.Event{{Type: core.EventTextDelta, Text: "done"}}},
	}}
	eng := engine.NewEngine(provider, nil)
	if _, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native"}, manager, eng); err != nil {
		t.Fatal(err)
	}
	stream, err := eng.Stream(context.Background(), core.Request{
		EnableToolDiscovery: true,
		SessionID:           "commentary",
		Model:               "native-model",
		Messages:            []core.Message{core.UserText("perform the special action")},
		MaxTurns:            4,
	})
	if err != nil {
		t.Fatal(err)
	}
	drainNativeStream(t, stream)

	if len(provider.requests) != 3 {
		t.Fatalf("provider requests = %d, want 3 (the commentary turn ended the run)", len(provider.requests))
	}
	outputs := nativeDiscoveryOutputs(provider.requests[1].Messages)
	if len(outputs) != 1 || len(outputs[0].Tools) != 1 || outputs[0].Tools[0].Spec.Name != "federation__special_action" {
		t.Fatalf("second request discovery outputs = %#v", outputs)
	}
}

// A guessed or stale tool name (for example one remembered from before a
// restart) must come back as an empty result the model can recover from, not
// fail the whole run.
func TestNativeDiscoveryUnknownToolNameKeepsRunGoing(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true, turns: []nativeScriptTurn{
		{events: []core.Event{{Type: core.EventDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-1", Arguments: json.RawMessage(`{"tool_names":["mcp__github__pull_request_read"]}`)}}}},
		{events: []core.Event{{Type: core.EventDiscoveryCall, DiscoveryCall: &core.ToolDiscoveryCall{ID: "search-2", Arguments: json.RawMessage(`{"tool_names":["special_action"]}`)}}}},
		{events: []core.Event{{Type: core.EventToolCall, Tool: &core.ToolCall{ID: "external-1", Namespace: "federation", ChildName: "special_action", Arguments: json.RawMessage(`{"project_id":"P-42"}`)}}}},
		{events: []core.Event{{Type: core.EventTextDelta, Text: "done"}}},
	}}
	eng := engine.NewEngine(provider, nil)
	if _, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native"}, manager, eng); err != nil {
		t.Fatal(err)
	}
	stream, err := eng.Stream(context.Background(), core.Request{
		EnableToolDiscovery: true,
		SessionID:           "unknown-name",
		Model:               "native-model",
		Messages:            []core.Message{core.UserText("perform the special action")},
		MaxTurns:            6,
	})
	if err != nil {
		t.Fatal(err)
	}
	drainNativeStream(t, stream)

	if len(provider.requests) != 4 {
		t.Fatalf("provider requests = %d, want 4", len(provider.requests))
	}
	for _, output := range nativeDiscoveryOutputs(provider.requests[1].Messages) {
		if output.CallID == "search-1" && len(output.Tools) != 0 {
			t.Fatalf("unknown name loaded tools: %#v", output.Tools)
		}
	}
	outputs := nativeDiscoveryOutputs(provider.requests[2].Messages)
	if len(outputs) == 0 || outputs[len(outputs)-1].CallID != "search-2" || len(outputs[len(outputs)-1].Tools) != 1 || outputs[len(outputs)-1].Tools[0].Spec.Name != "federation__special_action" {
		t.Fatalf("retry discovery outputs = %#v", outputs)
	}
}

// A guessed name that shares words with a real tool finds it through a search
// on those words.
func TestNativeDiscoveryGuessedToolNameSearchesItsWords(t *testing.T) {
	manager := startDiscoveryTestManager(t)
	defer manager.StopAll()
	provider := &nativeScriptProvider{supported: true}
	eng := engine.NewEngine(provider, nil)
	planner, err := NewPlanner(config.ToolDiscoveryConfig{Mode: "deferred", Strategy: "native"}, manager, eng)
	if err != nil {
		t.Fatal(err)
	}
	req := core.Request{SessionID: "guessed", Model: "native-model"}
	if _, err := planner.BeginRun(context.Background(), provider, &req, "guessed-run"); err != nil {
		t.Fatal(err)
	}
	defer planner.EndRun("guessed-run")
	if _, err := planner.PrepareTurn(context.Background(), provider, &req, "guessed-run", 0, 5); err != nil {
		t.Fatal(err)
	}
	output, err := planner.ResolveNativeToolDiscovery(context.Background(), "guessed-run", core.ToolDiscoveryCall{ID: "search", Arguments: json.RawMessage(`{"tool_names":["mcp__federation__special_action_v2"]}`)})
	if err != nil {
		t.Fatal(err)
	}
	if !hasDiscoveredTool(output, "federation__special_action") {
		t.Fatalf("guessed-name output = %#v, want federation__special_action", output.Tools)
	}
}

func hasDiscoveredTool(output core.ToolDiscoveryOutput, name string) bool {
	for _, tool := range output.Tools {
		if tool.Spec.Name == name {
			return true
		}
	}
	return false
}
