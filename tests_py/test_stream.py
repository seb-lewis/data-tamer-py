import asyncio
import json

import pytest
litellm = pytest.importorskip("litellm")

import data_tamer.core as core_mod
from litellm import completion as ll_completion


def test_stream_transform_object_with_mock_response(monkeypatch):
    # Arrange: stream JSON via LiteLLM mock_response
    def _fake_completion(*, model, messages, temperature=0, response_format=None, stream=False, **kwargs):
        if stream:
            # LiteLLM supports stream=True + mock_response; return generator
            return ll_completion(model=model, messages=messages, stream=True, mock_response='{"a":1}', custom_llm_provider="openai")
        return ll_completion(model=model, messages=messages, mock_response='{"a":1}', custom_llm_provider="openai")

    monkeypatch.setattr(core_mod, "completion", _fake_completion)

    # Act: get the stream and collect chunks
    stream = core_mod.stream_transform_object(
        model="dummy-model",
        schema={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
        prompt_context={"instructions": "Return a JSON object with key 'a'"},
    )

    async def _consume():
        chunks = []
        async for ch in stream.object_stream:
            chunks.append(ch)
        obj = await stream.object()
        return "".join(chunks), obj

    chunks_str, obj = asyncio.run(_consume())
    assert isinstance(obj, dict)
    assert obj.get("a") == 1
    assert chunks_str
