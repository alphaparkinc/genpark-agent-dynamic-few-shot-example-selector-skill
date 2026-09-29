import sys, json, re, math
from collections import Counter

class AgentDynamicFewShotSelector:
    """
    Dynamic Few-Shot In-Context Exemplar Selector.
    Selects relevant and diverse demonstration examples from an example bank
    using Maximal Marginal Relevance (MMR) and character/word n-gram cosine similarity.
    """
    def __init__(self):
        self.examples = []

    def _tokenize(self, text):
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def _get_ngrams(self, text, n=3):
        cleaned = re.sub(r"\s+", " ", text.lower().strip())
        return Counter([cleaned[i:i+n] for i in range(max(0, len(cleaned) - n + 1))])

    def _cosine_similarity(self, counter_a, counter_b):
        if not counter_a or not counter_b:
            return 0.0
        intersection = sum(counter_a[k] * counter_b.get(k, 0) for k in counter_a)
        norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
        norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return intersection / (norm_a * norm_b)

    def add_example(self, example_id, input_data, output_data, metadata=None):
        text_repr = str(input_data)
        ngrams = self._get_ngrams(text_repr)
        self.examples.append({
            "id": example_id,
            "input": input_data,
            "output": output_data,
            "metadata": metadata or {},
            "ngrams": ngrams,
            "text_repr": text_repr
        })
        return {"status": "ADDED", "example_id": example_id, "bank_size": len(self.examples)}

    def select_few_shot_examples(self, query, k=3, method="mmr", lambda_mult=0.65):
        """
        Select k examples using Maximal Marginal Relevance (MMR):
        MMR = argmax [ lambda * Sim(query, ex) - (1 - lambda) * max_{s in selected} Sim(ex, s) ]
        """
        if not self.examples:
            return {"selected": [], "query": query, "total_bank": 0}

        q_ngrams = self._get_ngrams(query)
        candidates = list(self.examples)
        selected = []

        # Precompute query similarities
        query_sims = {ex["id"]: self._cosine_similarity(q_ngrams, ex["ngrams"]) for ex in candidates}

        k = min(k, len(candidates))
        while len(selected) < k and candidates:
            best_score = -999.0
            best_candidate = None
            best_idx = -1

            for idx, cand in enumerate(candidates):
                sim_to_query = query_sims[cand["id"]]
                if not selected:
                    mmr_score = sim_to_query
                else:
                    max_sim_to_selected = max(self._cosine_similarity(cand["ngrams"], s["ngrams"]) for s in selected)
                    mmr_score = (lambda_mult * sim_to_query) - ((1.0 - lambda_mult) * max_sim_to_selected)

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = cand
                    best_idx = idx

            if best_candidate:
                selected.append({
                    "id": best_candidate["id"],
                    "input": best_candidate["input"],
                    "output": best_candidate["output"],
                    "relevance_score": round(query_sims[best_candidate["id"]], 4),
                    "mmr_score": round(best_score, 4)
                })
                candidates.pop(best_idx)

        return {
            "query": query,
            "method": method,
            "selected_examples": selected,
            "selected_count": len(selected)
        }

    def format_prompt_with_examples(self, system_instruction, selected_examples, target_query):
        parts = [f"<system_instruction>
{system_instruction}
</system_instruction>
"]
        parts.append("<demonstrations>")
        for idx, ex in enumerate(selected_examples):
            parts.append(f"  <example index="{idx + 1}">")
            parts.append(f"    <input>{ex['input']}</input>")
            parts.append(f"    <output>{ex['output']}</output>")
            parts.append("  </example>")
        parts.append("</demonstrations>
")
        parts.append(f"<current_query>
{target_query}
</current_query>")
        return "
".join(parts)

    def run_selector_benchmark(self):
        self.examples.clear()
        # Populate diverse few-shot demonstrations
        self.add_example("ex_math_1", "Calculate 15% tip on $80 bill.", "Tip: $12.00 | Total: $92.00", {"type": "finance"})
        self.add_example("ex_math_2", "Compute compound interest on $1000 at 5% for 2 years.", "A = $1000 * (1 + 0.05)^2 = $1102.50", {"type": "finance"})
        self.add_example("ex_code_1", "Write python function to reverse a string.", "def rev(s): return s[::-1]", {"type": "coding"})
        self.add_example("ex_code_2", "Write python function to check if word is palindrome.", "def is_pal(w): return w.lower() == w[::-1].lower()", {"type": "coding"})
        self.add_example("ex_sql_1", "Query all active users registered after Jan 2026.", "SELECT * FROM users WHERE status = 'active' AND created_at > '2026-01-01';", {"type": "database"})

        query = "Calculate a 20% discount on a $250 purchase."
        selection = self.select_few_shot_examples(query, k=2)
        prompt = self.format_prompt_with_examples("You are an expert mathematical assistant.", selection["selected_examples"], query)

        return {
            "suite": "Dynamic Few-Shot In-Context Selector Benchmark",
            "query": query,
            "selection": selection,
            "compiled_prompt_sample": prompt[:300] + "...",
            "status": "OPTIMAL_SELECTION_ACTIVE"
        }
