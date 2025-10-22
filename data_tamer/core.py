from __future__ import annotations

import time
from typing import Any, Optional, Dict, List

try:
    import litellm
    from litellm import completion, supports_response_schema
except Exception as e:  # pragma: no cover - import-time guard
    litellm = None  # type: ignore
    completion = None  # type: ignore
    supports_response_schema = None  # type: ignore

from .prompt import build_batch_prompt, build_single_prompt
from .types import PromptBuilderOptions, PromptContext, TransformObjectResult

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore


def _ensure_litellm():
    if completion is None:
        raise RuntimeError(
            "litellm is not installed. Install with `uv add litellm` or `pip install litellm`."
        )


def _messages_for_prompt(prompt: str, system: Optional[str] = None) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _extract_text(resp: Any) -> str:
    # Try OpenAI-like response structure
    for path in (
        lambda r: r["choices"][0]["message"]["content"],
        lambda r: r["choices"][0]["text"],
        lambda r: r.choices[0].message["content"],
        lambda r: r.choices[0].message.content,
        lambda r: r.choices[0].text,
    ):
        try:
            return path(resp)
        except Exception:
            pass
    return str(resp)


def _parse_json_object(text: str) -> Any:
    import json

    try:
        return json.loads(text)
    except Exception:
        # Try to find a JSON object/array span
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
        raise


def _response_format_for_schema(model: Any, schema: Any) -> Any:
    """Return a LiteLLM-compatible response_format for a given schema.

    - Pydantic BaseModel subclass: return the class directly
    - JSON Schema dict: return {"type":"json_schema","json_schema": schema, "strict": True}
    - None: return {"type":"json_object"}
    """
    if schema is None:
        return {"type": "json_object"}

    try:
        from pydantic import BaseModel as _BM  # type: ignore

        if isinstance(schema, type) and issubclass(schema, _BM):
            return schema
    except Exception:
        pass

    if isinstance(schema, dict):
        return {"type": "json_schema", "json_schema": schema, "strict": True}

    # Fallback to generic JSON
    return {"type": "json_object"}


def transform_object(
    *,
    model: Any,
    schema: Any | None = None,
    items: Optional[list[Any]] = None,
    prompt_context: Optional[PromptContext] = None,
    prompt_options: Optional[PromptBuilderOptions] = None,
    debug: bool = False,
    max_retries: int = 1,
    repair: bool = True,
    # passthroughs held for API parity; not all are used directly by ai-sdk-python
    provider_options: Optional[dict] = None,
    max_tokens: Optional[int] = None,
) -> TransformObjectResult:
    """Generate a single structured object.

    Mirrors the TS API shape while using ai-sdk-python's `generate_object`.
    If `items` are provided, a compact batch-like prompt is built; otherwise a single-object prompt is built.
    """

    _ensure_litellm()

    has_items = bool(items)
    if not has_items and not prompt_context:
        raise ValueError("Provide items or prompt_context to build a prompt.")

    prompt = (
        build_batch_prompt(items=items or [], schema=schema, options=prompt_options, context=prompt_context)
        if has_items
        else build_single_prompt(schema=schema, options=prompt_options, context=prompt_context)
    )

    if debug:
        try:
            print("[data-tamer] Prompt (single/object)\n", prompt)
        except Exception:
            pass

    attempt = 0
    while True:
        try:
            # Use LiteLLM completion, prefer JSON object response format when supported.
            system_content = None  # prompt already includes guidance; keep simple
            msgs = _messages_for_prompt(prompt, system=system_content)
            # Choose best response_format based on provided schema
            rf = _response_format_for_schema(model, schema)
            # Enable client-side validation when using json_schema
            if litellm is not None and isinstance(rf, dict) and rf.get("type") == "json_schema":
                try:
                    litellm.enable_json_schema_validation = True  # type: ignore[attr-defined]
                except Exception:
                    pass
            try:
                resp = completion(model=model, messages=msgs, temperature=0, response_format=rf)
            except Exception:
                # Fallback: try without response_format
                resp = completion(model=model, messages=msgs, temperature=0)

            text = _extract_text(resp)
            value = _parse_json_object(text)
            # Local JSON Schema validation when applicable
            if isinstance(schema, dict) and jsonschema is not None:
                jsonschema.validate(instance=value, schema=schema)  # type: ignore[attr-defined]

            # Return raw text as response for debugging
            return {"data": value, "response": text}
        except Exception as err:
            attempt += 1
            if attempt > max_retries:
                raise
            # simple backoff; `repair` is advisory since ai-sdk-python may not expose repair hooks
            if debug:
                print(f"[data-tamer] Error attempt {attempt}: {err}")
            if not repair:
                time.sleep(min(1.0 * attempt, 3.0))
                continue
            time.sleep(min(1.0 * attempt, 3.0))


def stream_transform_object(
    *,
    model: Any,
    schema: Any | None = None,
    items: Optional[list[Any]] = None,
    prompt_context: Optional[PromptContext] = None,
    prompt_options: Optional[PromptBuilderOptions] = None,
    debug: bool = False,
    # passthroughs for API parity
    provider_options: Optional[dict] = None,
    max_tokens: Optional[int] = None,
):
    """Stream a structured object using ai-sdk-python's `stream_object`.

    Returns the SDK's stream result object (with `.object_stream` and `.object()`), mirroring TS behavior.
    """

    _ensure_litellm()

    has_items = bool(items)
    if not has_items and not prompt_context:
        raise ValueError("Provide items or prompt_context to build a prompt.")

    prompt = (
        build_batch_prompt(items=items or [], schema=schema, options=prompt_options, context=prompt_context)
        if has_items
        else build_single_prompt(schema=schema, options=prompt_options, context=prompt_context)
    )

    if debug:
        try:
            print("[data-tamer] Prompt (single/stream)\n", prompt)
        except Exception:
            pass

    # Streaming wrapper that yields text chunks and can parse final object
    import asyncio

    class LiteLLMStream:
        def __init__(self, model: Any, messages: List[Dict[str, str]], comp_func):
            self._model = model
            self._messages = messages
            self._buffer: List[str] = []
            self._comp = comp_func

        async def object(self) -> Any:
            text = "".join(self._buffer)
            return _parse_json_object(text)

        @property
        def object_stream(self):  # async generator of text
            async def gen():
                nonlocal prompt
                try:
                    # litellm streaming is sync generator; run in thread to avoid blocking

                    def _iter():
                        try:
                            for chunk in self._comp(model=self._model, messages=self._messages, stream=True, temperature=0):
                                yield chunk
                        except Exception as e:  # pragma: no cover
                            raise e

                    loop = asyncio.get_running_loop()

                    def _collect_chunks():
                        for ch in _iter():
                            yield ch

                    # consume sync generator in thread, push deltas
                    from queue import Queue
                    import threading

                    q: Queue = Queue()

                    def _worker():
                        try:
                            for ch in _collect_chunks():
                                q.put(ch)
                        finally:
                            q.put(None)

                    t = threading.Thread(target=_worker, daemon=True)
                    t.start()
                    while True:
                        item = await loop.run_in_executor(None, q.get)
                        if item is None:
                            break
                        # extract delta text
                        delta_txt = None
                        try:
                            delta_txt = item["choices"][0]["delta"].get("content")
                        except Exception:
                            try:
                                delta_txt = item.choices[0].delta.get("content")
                            except Exception:
                                delta_txt = None
                        if delta_txt:
                            self._buffer.append(delta_txt)
                            yield delta_txt
                except Exception:
                    return

            return gen()

    msgs = _messages_for_prompt(prompt)
    return LiteLLMStream(model, msgs, completion)
