"""Turn a Pydantic model into a JSON schema a small model will actually fill in.

Why this module exists
----------------------
Pydantic marks any field carrying a default as *not* required. Ollama compiles the
schema into a GBNF grammar, and an optional property is one the grammar lets the model
skip. A 3B model skips them constantly.

Measured on ``llama3.2:3b`` with the raw Pydantic schema for ``ExtractionResult``:

* every one of the six ``evidence[].value`` entries came back empty - the model emitted
  ``field`` and ``source_quote`` and omitted the value, which silently defeats every
  grounding check downstream;
* ``date_of_loss``, ``claim_number`` and ``reported_amount`` were dropped from ``fields``
  altogether, even though the model had plainly located them (they appear verbatim in
  the evidence quotes) - it simply had nowhere mandatory to put them.

Hardening the schema - every property required, no defaults, no extra properties - turns
"the model may answer" into "the model must answer". Absence is then expressed *in the
data* as ``""`` or ``[]``, which is what the analyzer expects, rather than as a missing
key that is indistinguishable from the model not bothering.

This is a deliberate trade: we accept a slightly harder generation task in exchange for
output whose shape we can rely on.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for ``model`` with every property mandatory.

    The input model is untouched; the returned dict is a deep copy safe to mutate.
    """
    schema = copy.deepcopy(model.model_json_schema())
    defs = schema.get("$defs", {})
    _harden(schema)
    for definition in defs.values():
        _harden(definition)
    return schema


def _harden(node: Any) -> None:
    """Recursively force every object node to require all of its properties."""
    if not isinstance(node, dict):
        return

    # A default tells the model the key is skippable; strip it everywhere.
    node.pop("default", None)

    properties = node.get("properties")
    if isinstance(properties, dict) and properties:
        node["required"] = list(properties.keys())
        node["additionalProperties"] = False
        for subschema in properties.values():
            _harden(subschema)

    items = node.get("items")
    if isinstance(items, dict):
        _harden(items)

    # anyOf / oneOf / allOf branches (e.g. Optional[...]) still need cleaning.
    for key in ("anyOf", "oneOf", "allOf"):
        branches = node.get(key)
        if isinstance(branches, list):
            for branch in branches:
                _harden(branch)


def describe(schema: dict[str, Any]) -> str:
    """Compact human-readable rendering, used in prompts as a belt-and-braces hint.

    Grammar constraints already guarantee the shape, but small models produce better
    *content* when the expected keys are also stated in the prompt text.
    """
    defs = schema.get("$defs", {})
    lines: list[str] = []

    def walk(node: dict[str, Any], indent: int, seen: frozenset[str]) -> None:
        pad = "  " * indent
        for name, spec in (node.get("properties") or {}).items():
            spec = _resolve(spec, defs)
            desc = spec.get("description", "")
            enum = spec.get("enum")
            if enum:
                lines.append(f"{pad}{name}: one of {enum}" + (f"  # {desc}" if desc else ""))
            elif spec.get("type") == "array":
                item = _resolve(spec.get("items", {}), defs)
                if item.get("properties"):
                    lines.append(f"{pad}{name}: list of objects" + (f"  # {desc}" if desc else ""))
                    walk(item, indent + 1, seen)
                else:
                    item_enum = item.get("enum")
                    kind = f"one of {item_enum}" if item_enum else item.get("type", "string")
                    lines.append(f"{pad}{name}: list of {kind}" + (f"  # {desc}" if desc else ""))
            elif spec.get("properties"):
                lines.append(f"{pad}{name}: object" + (f"  # {desc}" if desc else ""))
                walk(spec, indent + 1, seen)
            else:
                lines.append(f"{pad}{name}: {spec.get('type', 'string')}" + (f"  # {desc}" if desc else ""))

    walk(schema, 0, frozenset())
    return "\n".join(lines)


def _resolve(spec: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Follow a local ``$ref`` one hop into ``$defs``."""
    ref = spec.get("$ref")
    if not ref or not ref.startswith("#/$defs/"):
        return spec
    target = defs.get(ref.split("/")[-1], {})
    merged = dict(target)
    for key, value in spec.items():
        if key != "$ref":
            merged[key] = value
    return merged
