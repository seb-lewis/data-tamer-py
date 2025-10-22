from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional

from .types import InputItem, PromptBuilderOptions, PromptContext


def _truncate(text: str, max_len: Optional[int]) -> str:
    if not max_len or len(text) <= max_len:
        return text
    # Reserve 1 char for ellipsis to mirror TS behavior
    return text[: max(0, max_len - 3)] + "…"


def _format_item(item: InputItem, char_limit: Optional[int]) -> str:
    if isinstance(item, str):
        return _truncate(item, char_limit)
    try:
        import json

        return _truncate(json.dumps(item, separators=(",", ":")), char_limit)
    except Exception:
        return _truncate(str(item), char_limit)


def build_system_prompt(ctx: Optional[PromptContext]) -> Optional[str]:
    if not ctx or not ctx.get("system"):
        return None
    return ctx["system"].strip()


def build_instruction_prompt(ctx: Optional[PromptContext]) -> Optional[str]:
    if not ctx or not ctx.get("instructions"):
        return None
    return ctx["instructions"].strip()


def build_examples_prompt(
    examples: Optional[List[Mapping[str, Any]]],
    char_limit: Optional[int] = None,
) -> Optional[str]:
    if not examples:
        return None
    lines: List[str] = []
    for ex in examples:
        inp = ex.get("input")
        lines.append("- input: " + _format_item(inp, char_limit))
        if "output" in ex:
            try:
                import json

                lines.append("  output: " + json.dumps(ex["output"]))
            except Exception:
                lines.append("  output: [unserializable]")
    return "Examples (compact):\n" + "\n".join(lines)


def build_items_prompt(items: List[InputItem], char_limit: Optional[int] = None) -> str:
    lines = [f"#{i}: {_format_item(item, char_limit)}" for i, item in enumerate(items)]
    return f"Inputs ({len(items)}):\n" + "\n".join(lines)


def build_batch_prompt(
    *,
    items: List[InputItem],
    schema: Any | None = None,  # unused here, kept for API parity
    options: Optional[PromptBuilderOptions] = None,
    context: Optional[PromptContext] = None,
) -> str:
    parts: List[str] = []

    system = build_system_prompt(context)
    if system:
        parts.append(system)

    instructions = build_instruction_prompt(context) or (
        "You are a data transformation engine. Produce strictly valid JSON. No commentary."
    )
    parts.append(instructions)

    examples = build_examples_prompt(
        context.get("examples") if context else None,
        (options or {}).get("char_limit_per_item") if options else None,
    )
    if examples:
        parts.append(examples)

    items_block = build_items_prompt(items, (options or {}).get("char_limit_per_item") if options else None)
    parts.append(items_block)

    parts.append(
        "Output: For each input, return a corresponding JSON element in order. No extra text."
    )

    return "\n\n".join(parts)


def build_single_prompt(
    *,
    schema: Any | None = None,  # unused here, kept for API parity
    options: Optional[PromptBuilderOptions] = None,
    context: Optional[PromptContext] = None,
) -> str:
    parts: List[str] = []

    system = build_system_prompt(context)
    if system:
        parts.append(system)

    instructions = build_instruction_prompt(context) or (
        "You are a data transformation engine. Produce strictly valid JSON. No commentary."
    )
    parts.append(instructions)

    examples = build_examples_prompt(
        context.get("examples") if context else None,
        (options or {}).get("char_limit_per_item") if options else None,
    )
    if examples:
        parts.append(examples)

    parts.append("Output: Return a single JSON object only.")

    return "\n\n".join(parts)

