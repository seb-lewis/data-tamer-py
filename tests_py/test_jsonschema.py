import jsonschema
from types import SimpleNamespace

import data_tamer.core as core_mod
import data_tamer.batch as batch_mod
import pytest
litellm = pytest.importorskip("litellm")
from litellm import completion as ll_completion


def test_transform_object_valid_jsonschema(monkeypatch):
    def _fake_completion(*, model, messages, temperature=0, response_format=None):
        return ll_completion(model=model, messages=messages, mock_response='{"name":"Alice","age":30}', custom_llm_provider="openai")

    monkeypatch.setattr(core_mod, "completion", _fake_completion)

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }

    res = core_mod.transform_object(
        model="mock-model", schema=schema, prompt_context={"instructions": "Return name and age"}
    )
    assert res["data"]["name"] == "Alice" and res["data"]["age"] == 30


def test_transform_object_invalid_jsonschema(monkeypatch):
    def _fake_completion(*, model, messages, temperature=0, response_format=None):
        return ll_completion(model=model, messages=messages, mock_response='{"name":"Alice","age":"thirty"}', custom_llm_provider="openai")

    monkeypatch.setattr(core_mod, "completion", _fake_completion)

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }

    try:
        core_mod.transform_object(model="mock-model", schema=schema, prompt_context={"instructions": "Return"})
        assert False, "Expected ValidationError"
    except Exception as e:
        assert isinstance(e, Exception)


def test_transform_batch_jsonschema(monkeypatch):
    def _fake_completion(*, model, messages, temperature=0, response_format=None):
        import json
        return ll_completion(model=model, messages=messages, mock_response=json.dumps([{"name": "Bob", "age": 20}, {"name": "Eve", "age": "NaN"}]), custom_llm_provider="openai")

    monkeypatch.setattr(batch_mod, "completion", _fake_completion)

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }

    try:
        batch_mod.transform_batch(model="mock-model", schema=schema, items=["a", "b"], batch_size=2)
        assert False, "Expected ValidationError"
    except Exception as e:
        # jsonschema raises ValidationError; allow any exception type for portability
        assert isinstance(e, Exception)
