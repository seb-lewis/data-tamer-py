import asyncio
import re
from types import SimpleNamespace

import data_tamer.batch as batch_mod
import pytest
litellm = pytest.importorskip("litellm")
from litellm import completion as ll_completion


def _fake_completion(*, model, messages, temperature=0, response_format=None):
    # messages[-1]['content'] contains our prompt
    prompt = messages[-1]["content"]
    m = re.search(r"Inputs \((\d+)\):", prompt)
    n = int(m.group(1)) if m else 1
    payload = [{"ok": True} for _ in range(n)]
    import json

    # Use LiteLLM mock_response to build a realistic response object
    return ll_completion(model=model, messages=messages, mock_response=json.dumps(payload), custom_llm_provider="openai")


def test_async_transform_batch_uses_concurrency(monkeypatch):
    monkeypatch.setattr(batch_mod, "completion", _fake_completion)

    items = [f"item {i}" for i in range(7)]
    results = asyncio.run(
        batch_mod.async_transform_batch(
            model="mock-model",
            schema=None,
            items=items,
            batch_size=3,
            concurrency=2,
            prompt_context={"instructions": "Return {\"ok\": true} per input"},
        )
    )

    # Should flatten to 7 dicts
    assert len(results) == 7
    assert all(isinstance(r, dict) and r.get("ok") is True for r in results)


@pytest.mark.asyncio
async def test_async_transform_batch_is_awaitable(monkeypatch):
    # Ensure we can await the coroutine directly under pytest asyncio
    monkeypatch.setattr(batch_mod, "completion", _fake_completion)

    items = [f"item {i}" for i in range(5)]
    results = await batch_mod.async_transform_batch(
        model="mock-model",
        schema=None,
        items=items,
        batch_size=2,
        concurrency=3,
        prompt_context={"instructions": "Return {\"ok\": true} per input"},
    )

    assert len(results) == 5
    assert all(isinstance(r, dict) and r.get("ok") is True for r in results)
