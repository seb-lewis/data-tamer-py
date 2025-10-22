from __future__ import annotations

from typing import Any, Dict, Type

try:
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover
    BaseModel = object  # type: ignore


def pydantic_json_schema(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Generate a JSON Schema dict from a Pydantic model class.

    Supports both Pydantic v2 (model_json_schema) and v1 (schema).
    """
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
        raise TypeError("pydantic_json_schema expects a Pydantic BaseModel subclass")

    # Pydantic v2
    if hasattr(model_cls, "model_json_schema"):
        return model_cls.model_json_schema()  # type: ignore[attr-defined]

    # Pydantic v1 fallback
    if hasattr(model_cls, "schema"):
        return model_cls.schema()  # type: ignore[attr-defined]

    raise RuntimeError("Unsupported Pydantic version or invalid model class")


def pydantic_array_json_schema(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Generate a JSON Schema for an array of a Pydantic model class."""
    return {"type": "array", "items": pydantic_json_schema(model_cls)}

