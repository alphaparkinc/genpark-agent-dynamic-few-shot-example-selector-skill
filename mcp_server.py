import sys, json
from client import AgentDynamicFewShotSelector

def handle_mcp():
    selector = AgentDynamicFewShotSelector()
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print(json.dumps(selector.run_selector_benchmark(), indent=2))
        return

    for line in sys.stdin:
        if not line.strip(): continue
        try:
            req = json.loads(line)
            method = req.get("method")
            msg_id = req.get("id")
            
            if method == "initialize":
                resp = {"jsonrpc": "2.0", "id": msg_id, "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "genpark-agent-dynamic-few-shot-example-selector-skill", "version": "1.0.0"},
                    "capabilities": {"tools": {}}
                }}
            elif method == "tools/list":
                resp = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": [
                    {"name": "add_example", "description": "Add exemplar to few-shot bank.", "inputSchema": {"type": "object", "properties": {"example_id": {"type": "string"}, "input_data": {"type": "string"}, "output_data": {"type": "string"}}}},
                    {"name": "select_few_shot_examples", "description": "Select top-k diverse examples via MMR.", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}}},
                    {"name": "run_selector_benchmark", "description": "Run few-shot selection benchmark.", "inputSchema": {"type": "object"}}
                ]}}
            elif method == "tools/call":
                tname = req.get("params", {}).get("name")
                args = req.get("params", {}).get("arguments", {})
                if tname == "add_example":
                    res = selector.add_example(args.get("example_id", "ex"), args.get("input_data", ""), args.get("output_data", ""))
                elif tname == "select_few_shot_examples":
                    res = selector.select_few_shot_examples(args.get("query", ""), args.get("k", 3))
                else:
                    res = selector.run_selector_benchmark()
                resp = {"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
            else:
                resp = {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "Method not found"}}
            
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}}) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    handle_mcp()
