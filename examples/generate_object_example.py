import os
from pydantic import BaseModel
from data_tamer import transform_object


class Person(BaseModel):
    name: str
    age: int | None


def main() -> None:
    model = os.environ.get("LITELLM_MODEL", "gpt-4o-mini")

    result = transform_object(
        model=model,
        schema=Person,
        prompt_context={
            "instructions": "Extract name and age. Use null when unknown.",
            "examples": [
                {"input": "John Doe, 34", "output": {"name": "John Doe", "age": 34}},
            ],
        },
    )

    print("Result:", result["data"])


if __name__ == "__main__":
    main()
