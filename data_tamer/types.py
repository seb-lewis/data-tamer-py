from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TypedDict, Union

InputItem = Union[str, Mapping[str, Any]]


class PromptContext(TypedDict, total=False):
    system: str
    instructions: str
    examples: List[Mapping[str, Any]]  # each example: { input: InputItem, output?: Any }


class PromptOptions(TypedDict, total=False):
    char_limit_per_item: int
    include_schema_description: bool


class PromptBuilderOptions(PromptOptions, total=False):
    schema_name: str
    schema_description: str


# Return type for transform_object to mirror TS shape
class TransformObjectResult(TypedDict, total=False):
    data: Any
    response: Any

