from client import AgentDynamicFewShotSelector
import json

selector = AgentDynamicFewShotSelector()
print("=== AGENT DYNAMIC FEW-SHOT SELECTOR BENCHMARK ===")
res = selector.run_selector_benchmark()
print(json.dumps(res, indent=2))
