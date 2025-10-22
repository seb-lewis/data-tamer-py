"""
Simple example: legacy contact cleanup (no CSV parsing)

Usage:
  python examples/legacy_contacts.py

Requirements:
  - Install: `pip install litellm pydantic` (or `uv add litellm pydantic`)
  - Set `OPENROUTER_API_KEY` in your environment
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from pydantic import BaseModel

from data_tamer import transform_batch


# Domain prompt provided by the user
PROMPT = "\n".join(
    [
        "- Remove honorifics/credentials (Dr., MD, DO, PhD, etc.) from the data",
        "- Always extract the requested fields for the primary person from the available data. If it is not clear, leave fields null.",
        "- Put information not related to the primary person in the notes field.",
        "- If no primary person name is present, leave name fields null.",
        "- notes are only for extra information that exists in the record but does not fit into the requested output fields, not for extra commentary or explanation.",
    ]
)


# Output schema
class LegacyContact(BaseModel):
    first_name: Optional[str]
    middle_name: Optional[str]
    last_name: Optional[str]
    title: Optional[str]
    department: Optional[str]
    address: Optional[str]
    notes: Optional[str]


rows: List[Dict[str, Any]] = [
    {
        "legacy_first_name": None,
        "legacy_middle_name": None,
        "legacy_last_name": "Ava Parker",
        "name_descriptor": None,
        "legacy_title": None,
        "address_text": None,
        "company": None,
    },
    {
        "legacy_first_name": None,
        "legacy_middle_name": None,
        "legacy_last_name": "jordan morgan",
        "name_descriptor": "sec taylor brooks",
        "legacy_title": None,
        "address_text": "1234 North Example Street",
        "company": None,
    },
    {
        "legacy_first_name": None,
        "legacy_middle_name": None,
        "legacy_last_name": "Dr. Sam Carter",
        "name_descriptor": None,
        "legacy_title": None,
        "address_text": None,
        "company": None,
    },
    {
        "legacy_first_name": None,
        "legacy_middle_name": None,
        "legacy_last_name": "Robert Monroe  Attention Morgan",
        "name_descriptor": None,
        "legacy_title": None,
        "address_text": None,
        "company": "Acme Health Systems",
    },
    {
        "legacy_first_name": None,
        "legacy_middle_name": None,
        "legacy_last_name": "Alex",
        "name_descriptor": None,
        "legacy_title": None,
        "address_text": None,
        "company": None,
    },
    {
        "legacy_first_name": None,
        "legacy_middle_name": None,
        "legacy_last_name": "Taylor (assistant)/Jordan",
        "name_descriptor": None,
        "legacy_title": None,
        # Include some inline labels to mirror real-world address_text
        "address_text": "1000 Sample Ave., #101\nEmail: sample.person@example.com\nPhone: 555-0100",
        "company": None,
    },
]


def _compact_row_values(r: Dict[str, Any]) -> str:
    return ",".join(str(v) for v in r.values() if v)


def main() -> None:
    # Use OpenRouter via LiteLLM
    import os

    # Example OpenRouter models: "openrouter/google/gemini-2.5-flash-lite", "openrouter/anthropic/claude-3.5-sonnet"
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/google/gemini-2.5-flash-lite")

    results = transform_batch(
        items=[_compact_row_values(r) for r in rows],
        schema=LegacyContact,
        model=model,
        batch_size=6,
        concurrency=1,
        prompt_context={
            "instructions": PROMPT,
        },
        debug=True,
    )

    print("\nTransformed rows:\n")
    for i, r in enumerate(results, start=1):
        try:
            import json

            print(f"#{i}", json.dumps(r, indent=2))
        except Exception:
            print(f"#{i}", r)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import sys

        print("Error:", getattr(err, "message", repr(err)))
        sys.exit(1)
