"""Single source of truth for task-specific LLM instructions."""


def build_prompt(*, task_type: str, query: str, context: str = "") -> str:
    """Fill the template for ``task_type`` ("qa", "summarization", "conversation").

    With no context the bare query is returned unchanged. The summarization
    template does not use the query. Raises ValueError for an unknown task_type.
    """
    templates = {
        "qa": (
            "{label}: {context}\n\nQuestion: {query}\n\n"
            "Answer using only the provided context. Reply with just the answer "
            "in as few words as possible, with no explanation."
        ),
        "summarization": (
            "Text: {context}\n\n"
            "Summarize the text faithfully and concisely. Do not add information."
        ),
        "conversation": (
            "Conversation context: {context}\n\nQuestion: {query}\n\n"
            "Answer using only the conversation context. Reply with just the answer "
            "in as few words as possible, with no explanation."
        ),
    }
    if task_type not in templates:
        raise ValueError(f"Unknown task_type: {task_type}")
    if not context:
        return query
    label = "Context"
    return templates[task_type].format(label=label, context=context, query=query)