from __future__ import annotations

import time
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Type
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from pydantic import BaseModel
    from pydantic import RootModel  # pydantic v2
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
    RootModel = None  # type: ignore

try:
    import litellm
    from litellm import completion
except Exception:  # pragma: no cover
    litellm = None  # type: ignore
    completion = None  # type: ignore

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore

from .prompt import build_batch_prompt
from .core import _messages_for_prompt, _extract_text, _parse_json_object
from .types import InputItem, PromptBuilderOptions, PromptContext


def _ensure_litellm():
    if completion is None:
        raise RuntimeError(
            "litellm is not installed. Install with `uv add litellm` or `pip install litellm`."
        )


def _make_list_schema(schema: Any) -> Any:
    """Attempt to wrap a Pydantic model class into a list schema for strict array output.

    - For Pydantic v2, create a `RootModel[List[Schema]]`
    - For Pydantic v1, fall back to dynamic __root__ model if available
    If schema is None or not a Pydantic model class, return None to indicate no strict schema.
    """
    if schema is None:
        return None

    # Heuristic check: treat classes inheriting BaseModel as model classes
    try:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            if RootModel is not None:
                from typing import List as TList

                # Create an inline RootModel[List[schema]]
                class _BatchModel(RootModel[TList[schema]]):  # type: ignore[name-defined]
                    pass

                return _BatchModel
            else:
                # pydantic v1 fallback via create_model with __root__
                from pydantic import create_model
                from typing import List as TList

                return create_model("BatchModel", __root__=(TList[schema], ...))  # type: ignore[name-defined]
    except Exception:
        return None

    # If user provides a JSON Schema dict, wrap it into an array schema
    if isinstance(schema, dict):
        return {"type": "array", "items": schema}

    return None


def transform_batch(
    *,
    model: Any,
    schema: Any | None,
    items: Sequence[InputItem],
    batch_size: int = 10,
    concurrency: int = 2,  # currently sequential; param retained for API parity
    provider_options: Optional[dict] = None,
    schema_name: Optional[str] = None,
    schema_description: Optional[str] = None,
    prompt_context: Optional[PromptContext] = None,
    prompt_options: Optional[PromptBuilderOptions] = None,
    max_retries: int = 1,
    repair: bool = True,
    on_batch_result: Optional[Callable[[dict], None]] = None,
    max_tokens: Optional[int] = None,
    debug: bool = False,
) -> List[Any]:
    """Transform a sequence of inputs in batches into structured outputs.

    Notes:
    - Uses a compact batch prompt per group of items to reduce tokens.
    - Attempts strict validation by wrapping the provided `schema` into a list schema when possible.
    - Executes batches using threads when `concurrency > 1`; otherwise sequentially.
    """

    _ensure_litellm()

    if not items:
        return []

    # Chunk items
    batches: List[List[InputItem]] = [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]

    results: List[List[Any]] = [[] for _ in range(len(batches))]

    def process_one(batch_index: int, batch_items: List[InputItem]) -> List[Any]:
        attempt = 0
        while True:
            try:
                if debug:
                    try:
                        print("[data-tamer] Batch size", {"batch_index": batch_index, "count": len(batch_items)})
                    except Exception:
                        pass

                prompt = build_batch_prompt(
                    items=batch_items,
                    schema=schema,
                    options={
                        **(prompt_options or {}),
                        **({"schema_name": schema_name} if schema_name else {}),
                        **({"schema_description": schema_description} if schema_description else {}),
                    },
                    context=prompt_context,
                )

                if debug:
                    try:
                        print("[data-tamer] Prompt (batch)", {"batch_index": batch_index, "prompt": prompt})
                    except Exception:
                        pass

                list_schema = _make_list_schema(schema)
                # Call LiteLLM and parse JSON
                rf = None
                if isinstance(list_schema, dict):
                    rf = {"type": "json_schema", "json_schema": list_schema, "strict": True}
                    if litellm is not None:
                        try:
                            litellm.enable_json_schema_validation = True  # type: ignore[attr-defined]
                        except Exception:
                            pass
                else:
                    rf = {"type": "json_object"}
                msgs = _messages_for_prompt(prompt)
                try:
                    resp = completion(model=model, messages=msgs, temperature=0, response_format=rf)
                except Exception:
                    resp = completion(model=model, messages=msgs, temperature=0)

                text = _extract_text(resp)
                obj = _parse_json_object(text)
                arr = obj if isinstance(obj, list) else [obj]

                if isinstance(list_schema, dict) and jsonschema is not None:
                    jsonschema.validate(instance=arr, schema=list_schema)  # type: ignore[attr-defined]

                return arr
            except Exception as err:
                attempt += 1
                if attempt > max_retries:
                    raise
                if debug:
                    print(f"[data-tamer] Batch error attempt {attempt} (batch {batch_index}): {err}")
                time.sleep(min(1.0 * attempt, 3.0))

    if concurrency and concurrency > 1:
        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            future_to_idx = {pool.submit(process_one, i, b): i for i, b in enumerate(batches)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
                if on_batch_result is not None:
                    on_batch_result({
                        "batch_index": idx,
                        "items": batches[idx],
                        "result": results[idx],
                    })
    else:
        for i, b in enumerate(batches):
            results[i] = process_one(i, b)
            if on_batch_result is not None:
                on_batch_result({
                    "batch_index": i,
                    "items": b,
                    "result": results[i],
                })

    return [item for group in results for item in group]


# Async variant with concurrency
import asyncio


async def async_transform_batch(
    *,
    model: Any,
    schema: Any | None,
    items: Sequence[InputItem],
    batch_size: int = 10,
    concurrency: int = 2,
    provider_options: Optional[dict] = None,
    schema_name: Optional[str] = None,
    schema_description: Optional[str] = None,
    prompt_context: Optional[PromptContext] = None,
    prompt_options: Optional[PromptBuilderOptions] = None,
    max_retries: int = 1,
    repair: bool = True,
    on_batch_result: Optional[Callable[[dict], None]] = None,
    max_tokens: Optional[int] = None,
    debug: bool = False,
) -> List[Any]:
    _ensure_litellm()

    if not items:
        return []

    batches: List[List[InputItem]] = [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]

    sem = asyncio.Semaphore(max(1, int(concurrency)))
    results: List[List[Any]] = [[] for _ in range(len(batches))]

    async def worker(batch_index: int, batch_items: List[InputItem]):
        nonlocal results
        async with sem:
            attempt = 0
            while True:
                try:
                    if debug:
                        try:
                            print("[data-tamer] Batch size", {"batch_index": batch_index, "count": len(batch_items)})
                        except Exception:
                            pass

                    prompt = build_batch_prompt(
                        items=batch_items,
                        schema=schema,
                        options={
                            **(prompt_options or {}),
                            **({"schema_name": schema_name} if schema_name else {}),
                            **({"schema_description": schema_description} if schema_description else {}),
                        },
                        context=prompt_context,
                    )

                    list_schema = _make_list_schema(schema)

                    # litellm completion is sync; run in default loop executor
                    loop = asyncio.get_running_loop()
                    rf = None
                    if isinstance(list_schema, dict):
                        rf = {"type": "json_schema", "json_schema": list_schema, "strict": True}
                        if litellm is not None:
                            try:
                                litellm.enable_json_schema_validation = True  # type: ignore[attr-defined]
                            except Exception:
                                pass
                    else:
                        rf = {"type": "json_object"}
                    try:
                        resp = await loop.run_in_executor(
                            None,
                            lambda: completion(model=model, messages=_messages_for_prompt(prompt), temperature=0, response_format=rf),
                        )
                    except Exception:
                        resp = await loop.run_in_executor(
                            None, lambda: completion(model=model, messages=_messages_for_prompt(prompt), temperature=0)
                        )

                    text = _extract_text(resp)
                    obj = _parse_json_object(text)
                    arr = obj if isinstance(obj, list) else [obj]

                    if isinstance(list_schema, dict) and jsonschema is not None:
                        jsonschema.validate(instance=arr, schema=list_schema)  # type: ignore[attr-defined]

                    results[batch_index] = arr
                    if on_batch_result is not None:
                        on_batch_result({
                            "batch_index": batch_index,
                            "items": batch_items,
                            "result": results[batch_index],
                        })
                    break
                except Exception as err:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    if debug:
                        print(f"[data-tamer] Batch error attempt {attempt} (batch {batch_index}): {err}")
                    await asyncio.sleep(min(1.0 * attempt, 3.0))

    await asyncio.gather(*(worker(i, b) for i, b in enumerate(batches)))
    return [item for group in results for item in group]
