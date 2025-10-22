import os
from typing import Optional
from pydantic import BaseModel
from data_tamer import transform_object, transform_batch, pydantic_json_schema


class Person(BaseModel):
    name: str
    age: Optional[int]


def main() -> None:
    model = os.environ.get("LITELLM_MODEL", "gpt-4o-mini")

    # Single object with JSON Schema validation
    single = transform_object(
        model=model,
        schema=pydantic_json_schema(Person),
        prompt_context={
            "instructions": "Extract name and age from the sentence. Use null when unknown.",
            "examples": [
                {"input": "John Doe, 34", "output": {"name": "John Doe", "age": 34}},
            ],
        },
    )
    print("single:", single["data"])

    # Batch with array-wrapped JSON Schema validation
    results = transform_batch(
        model=model,
        schema=pydantic_json_schema(Person),
        items=["Jane Doe, 29", "Mr. Smith, unknown age"],
        batch_size=2,
        prompt_context={"instructions": "Extract name and age; use null when unknown."},
    )
    print("batch:", results)


if __name__ == "__main__":
    main()
