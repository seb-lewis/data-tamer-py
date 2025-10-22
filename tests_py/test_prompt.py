import re

from data_tamer.prompt import (
    build_batch_prompt,
    build_single_prompt,
    build_examples_prompt,
    build_items_prompt,
)


def test_build_single_prompt_basic():
    prompt = build_single_prompt(context={"instructions": "Do X"})
    assert "Do X" in prompt
    assert "Return a single JSON object" in prompt


def test_build_batch_prompt_includes_inputs_count():
    p = build_batch_prompt(items=["a", "b"], schema=None, options=None, context=None)
    assert "Inputs (2):" in p
    assert "#0: a" in p and "#1: b" in p

