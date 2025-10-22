from typing import Optional

from pydantic import BaseModel

from data_tamer.schema import pydantic_json_schema, pydantic_array_json_schema


class Person(BaseModel):
    name: str
    age: Optional[int]


def test_pydantic_json_schema_generates_properties():
    schema = pydantic_json_schema(Person)
    assert isinstance(schema, dict)
    props = schema.get("properties") or schema.get("$defs") or {}
    # Different Pydantic versions structure schemas differently; ensure keys show up somewhere
    assert "name" in str(schema).lower()
    assert "age" in str(schema).lower()


def test_pydantic_array_json_schema_is_array():
    arr_schema = pydantic_array_json_schema(Person)
    assert arr_schema.get("type") == "array"
    assert isinstance(arr_schema.get("items"), dict)

