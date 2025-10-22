# data-tamer

Lightweight Python wrappers (built on LiteLLM) for transforming data with structured outputs, compact prompts for lower token usage, and batching utilities. Strict structured outputs are supported via Pydantic models or JSON Schema.

## Install

Install from PyPI via pip or UV:

```
pip install data-tamer
# or with UV
uv add data-tamer
```

Basic usage in Python mirrors the TS API and prompt-compaction behavior:

```python
from pydantic import BaseModel
import os
from data_tamer import transform_object, transform_batch


class Person(BaseModel):
    name: str
    age: int | None

# Choose a LiteLLM model id; set provider API keys via env (e.g., OPENAI_API_KEY, OPENROUTER_API_KEY)
model = os.environ.get("LITELLM_MODEL", "gpt-4o-mini")

# Single transform from guidance only
single = transform_object(
    model=model,
    schema=Person,
    prompt_context={
        "instructions": "Extract name and age. Use null when unknown.",
    },
)
print(single["data"])  # -> Person(name=..., age=...)

# Batch transform from compact prompt
inputs = [
    "Jane Doe, 29",
    "Mr. Smith, unknown age",
    {"text": "Alice, 41"},
]

results = transform_batch(
    model=model,
    schema=Person,
    items=inputs,
    batch_size=2,
    prompt_context={
        "instructions": "Extract name and age. Use null when unknown.",
    },
)
print(results)  # list of Person-like dicts
```

Streaming structured output is supported via `data_tamer.stream_transform_object` (LiteLLM streaming under the hood).

### Async batching

For higher throughput, use the async variant with concurrency:

```python
import asyncio
from pydantic import BaseModel
import os
from data_tamer import async_transform_batch


class Person(BaseModel):
    name: str
    age: int | None


async def main():
    model = os.environ.get("LITELLM_MODEL", "gpt-4o-mini")
    inputs = [f"User {i}, {20 + (i % 40)}" for i in range(100)]
    results = await async_transform_batch(
        model=model,
        schema=Person,
        items=inputs,
        batch_size=10,
        concurrency=5,
        prompt_context={"instructions": "Extract name and age"},
    )
    print(len(results))


asyncio.run(main())
```

## Prompt Compaction

The prompt builder:

- De-duplicates schema guidance and uses short, strict JSON directions.
- Truncates per-item input via `char_limit_per_item`.
- Supports optional `system`, `instructions`, and few-shot `examples`.
- Items are raw inputs (strings or objects). Place guidance/instructions in `prompt_context.system`/`prompt_context.instructions`.

## API

- `transform_object(model, schema, items|prompt_context, ...)`
  - Generates a single structured object. If `items` are provided, a compact prompt is built; otherwise use `prompt_context` with instructions.
  - `schema` can be a Pydantic model class or a JSON Schema dict. When supported by the provider, LiteLLM enforces structured output. We also parse JSON and, for dict schemas, validate locally via `jsonschema` as a fallback.

- `stream_transform_object(...)`
  - Streams text chunks and allows awaiting the final parsed object.

- `transform_batch(model, schema, items, batch_size=..., concurrency=...)`
  - Splits inputs into batches, builds compact prompts, and parses array outputs. Uses threads when `concurrency > 1`.

- `async_transform_batch(...)`
  - Async variant with concurrency control via asyncio.

## Notes

- Providers (LiteLLM): pass a model id string (e.g., `gpt-4o-mini`, `openrouter/google/gemini-2.5-flash-lite`) and set the corresponding API key in env (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, etc.).
- Structured outputs:
  - Pydantic: pass a `BaseModel` subclass as `schema`. LiteLLM will request structured responses when supported; we parse JSON regardless.
  - JSON Schema: pass a dict; we set LiteLLM `response_format={"type":"json_schema",...}` and also validate locally with `jsonschema`.
  - Helpers: `pydantic_json_schema`, `pydantic_array_json_schema` generate dict schemas from Pydantic models.
- OpenRouter: set `OPENROUTER_API_KEY` and pick an OpenRouter model id via `LITELLM_MODEL`, e.g., `openrouter/google/gemini-2.5-flash-lite`.

## Examples

- `examples/generate_object_example.py` — basic structured generation
- `examples/transform_batch_example.py` — batching with compact prompts
- `examples/jsonschema_example.py` — JSON Schema with validation
- `examples/legacy_contacts.py` — real-world cleanup with OpenRouter (default Gemini model)
