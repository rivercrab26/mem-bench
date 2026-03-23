"""Adapter registry and discovery."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from mem_bench.core.adapter import MemoryAdapter

_BUILTIN_ADAPTERS: dict[str, str] = {
    "bm25": "mem_bench.adapters.bm25:BM25Adapter",
    "mem0": "mem_bench.adapters.mem0:Mem0Adapter",
    "graphiti": "mem_bench.adapters.graphiti:GraphitiAdapter",
    "langmem": "mem_bench.adapters.langmem:LangMemAdapter",
    "letta": "mem_bench.adapters.letta:LettaAdapter",
    "supermemory": "mem_bench.adapters.supermemory:SupermemoryAdapter",
    "hindsight": "mem_bench.adapters.hindsight:HindsightAdapter",
}


def _import_class(dotted_path: str) -> type:
    module_path, class_name = dotted_path.rsplit(":", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_adapter(name: str, options: dict[str, Any] | None = None) -> MemoryAdapter:
    """Get an adapter instance by name.

    Resolution order:
    1. Built-in adapters (by short name)
    2. Entry points (from installed packages)
    3. Dynamic import (dotted.path:ClassName)
    """
    options = options or {}

    # 1. Built-in
    if name in _BUILTIN_ADAPTERS:
        cls = _import_class(_BUILTIN_ADAPTERS[name])
        return cls(**options)

    # 2. Entry points
    eps = entry_points()
    if hasattr(eps, "select"):
        group = eps.select(group="mem_bench.adapters")
    else:
        group = eps.get("mem_bench.adapters", [])
    for ep in group:
        if ep.name == name:
            cls = ep.load()
            return cls(**options)

    # 3. Dynamic import
    if ":" in name:
        cls = _import_class(name)
        return cls(**options)

    available = list(_BUILTIN_ADAPTERS.keys())
    raise ValueError(f"Unknown adapter '{name}'. Available: {available}")


def list_adapters() -> list[str]:
    """List all available adapter names."""
    names = list(_BUILTIN_ADAPTERS.keys())
    eps = entry_points()
    if hasattr(eps, "select"):
        group = eps.select(group="mem_bench.adapters")
    else:
        group = eps.get("mem_bench.adapters", [])
    for ep in group:
        if ep.name not in names:
            names.append(ep.name)
    return sorted(names)
