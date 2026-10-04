"""Budget-aware selection utilities for selective compression."""

from token_analysis.counter import count_tokens


class BudgetSelector:
    """Select high-scoring segments without exceeding a real token budget."""

    def __init__(self, model: str):
        self.model = model

    def select(self, segments, scores, budget_tokens: int):
        if not segments:
            return [], []
        if budget_tokens <= 0:
            budget_tokens = 1

        costs = [count_tokens(s.text, self.model) for s in segments]
        ranked = sorted(range(len(segments)), key=lambda i: (scores[i], -i), reverse=True)

        selected = []
        used = 0
        for i in ranked:
            cost = costs[i]
            if not selected and cost > budget_tokens:
                selected.append(i)
                used = cost
                break
            if used + cost <= budget_tokens:
                selected.append(i)
                used += cost

        # If nothing fit, keep the shortest segment. This satisfies the
        # compressor contract: never return an empty context.
        if not selected:
            selected = [min(range(len(segments)), key=lambda i: costs[i])]

        return sorted(selected), costs
