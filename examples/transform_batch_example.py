import os
from typing import Optional

from pydantic import BaseModel
from data_tamer import transform_batch


class Person(BaseModel):
    name: str
    age: Optional[int]


def main() -> None:
    model = os.environ.get("LITELLM_MODEL", "gpt-4o-mini")

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
        debug=True,
    )

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
