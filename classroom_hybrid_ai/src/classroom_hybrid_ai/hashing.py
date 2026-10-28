"""Small, dependency-free hashing helpers for provenance receipts."""

from __future__ import annotations

import hashlib
import json
import ast
import time
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator


_HASH_CACHE: ContextVar[dict[tuple[str, int, int, int], str] | None] = ContextVar("artifact_hash_cache", default=None)


@contextmanager
def hash_file_cache() -> Iterator[None]:
    """Share immutable file reads only for one producer's commit transaction."""

    token = _HASH_CACHE.set({})
    try:
        yield
    finally:
        _HASH_CACHE.reset(token)


@contextmanager
def uncached_hashes() -> Iterator[None]:
    """Recovery and independent audits always read file content again."""

    token = _HASH_CACHE.set(None)
    try:
        yield
    finally:
        _HASH_CACHE.reset(token)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    cache = _HASH_CACHE.get()
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    try:
        from .performance import increment, record_timing
    except ImportError:
        def increment(*args: Any) -> None:
            pass
        def record_timing(*args: Any) -> None:
            pass
    if cache is not None and key in cache:
        increment("hash.cache_hits")
        return cache[key]
    started = time.perf_counter()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    result = digest.hexdigest()
    record_timing("hash.seconds", time.perf_counter() - started)
    increment("hash.bytes_read", stat.st_size)
    if cache is not None:
        cache[key] = result
    return result


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def python_source_digest(package_root: str | Path | None = None) -> str:
    """Hash the installed package's Python sources with stable relative paths."""

    root = Path(package_root).resolve() if package_root is not None else Path(__file__).resolve().parent
    return canonical_sha256([
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*.py"))
        if path.is_file() and "__pycache__" not in path.parts
    ])


def project_config_payload(config: Any) -> dict[str, Any]:
    """Return the location-independent semantic payload for a ProjectConfig.

    The source checkout spells the repository root as ``..`` while the wheel
    resource uses ``@cwd``.  Once loaded, both values resolve to the same
    workspace.  Provenance receipts therefore hash the resolved root rather
    than the packaging-specific spelling, while preserving every other policy
    value verbatim.
    """

    payload = deepcopy(config.payload)
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        paths = {}
        payload["paths"] = paths
    paths["repository_root"] = Path(config.repository_root).resolve().as_posix()
    return payload


def project_config_digest(config: Any) -> str:
    """Hash a loaded project configuration independent of package layout."""

    return canonical_sha256(project_config_payload(config))


def stage_dependency_fingerprints(config: Any) -> dict[str, dict[str, str]]:
    """Hash executable symbol dependencies and the config consumed by each stage.

    The symbol closure splits the two legacy monoliths without copying code.
    Imports, assignments, helpers and class methods reached by a stage are
    included. Shared orchestration remains a conservative all-stage dependency.
    The upstream receipt chain propagates every earlier-stage invalidation.
    """

    root = Path(config.project_root) / "src" / "classroom_hybrid_ai"
    if not root.is_dir():
        # Wheel resources and user-supplied config directories have no src/
        # checkout. Hash the package actually executing in that case.
        root = Path(__file__).resolve().parent
    parsed: dict[str, tuple[dict[str, ast.AST], dict[str, tuple[str, str]]]] = {}

    def module_symbols(module: str) -> tuple[dict[str, ast.AST], dict[str, tuple[str, str]]]:
        if module in parsed:
            return parsed[module]
        path = root.joinpath(*module.split(".")).with_suffix(".py")
        is_package = False
        if not path.is_file():
            path = root.joinpath(*module.split("."), "__init__.py")
            is_package = True
        symbols: dict[str, ast.AST] = {}
        imports: dict[str, tuple[str, str]] = {}
        if path.is_file():
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols[node.name] = node
                    if isinstance(node, ast.ClassDef):
                        for child in node.body:
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                symbols[f"{node.name}.{child.name}"] = child
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            symbols[target.id] = node
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    parent = module.split(".") if is_package else module.split(".")[:-1]
                    base = parent[:len(parent) - node.level + 1]
                    if node.module:
                        base.extend(node.module.split("."))
                    for alias in node.names:
                        imports[alias.asname or alias.name] = (".".join(base), alias.name)
        parsed[module] = symbols, imports
        return symbols, imports

    def closure(seeds: list[tuple[str, str]]) -> dict[str, str]:
        result: dict[str, str] = {}
        seen: set[tuple[str, str]] = set()
        pending = list(seeds)
        while pending:
            module, name = pending.pop()
            if (module, name) in seen:
                continue
            seen.add((module, name))
            symbols, imports = module_symbols(module)
            if name in imports:
                pending.append(imports[name])
                continue
            node = symbols.get(name)
            if node is None:
                continue
            result[f"{module}:{name}"] = canonical_sha256(ast.dump(node, include_attributes=False))
            class_name = name.split(".")[0] if "." in name else None
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    if child.id in symbols or child.id in imports:
                        # The executor is rooted by method, never by the class
                        # containing all eight independent stage implementations.
                        if child.id != "ResearchStageExecutor":
                            pending.append((module, child.id))
                elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                    if child.value.id in {"self", "cls"} and class_name:
                        helper = f"{class_name}.{child.attr}"
                        if helper in symbols:
                            pending.append((module, helper))
                    elif child.value.id in imports:
                        target_module, target_name = imports[child.value.id]
                        pending.append((f"{target_module}.{target_name}".strip("."), child.attr))
        return result

    # These files define control flow, storage and the fingerprint algorithm
    # itself; any executable change is conservatively shared by all stages.
    shared = {
        name: sha256_file(root / name)
        for name in ("research_backend.py", "hashing.py", "storage.py", "contracts.py", "config.py")
        if (root / name).is_file()
    }
    for name in ("pyproject.toml",):
        path = Path(config.project_root) / name
        if path.is_file():
            shared[name] = sha256_file(path)
    for name in ("requirements.txt", "uv.lock", "poetry.lock"):
        path = Path(config.repository_root) / name
        if path.is_file():
            shared[name] = sha256_file(path)
    symbols, _ = module_symbols("research_stages")
    constructor = symbols.get("ResearchStageExecutor.__init__")
    if constructor is not None:
        shared["executor_initialization"] = canonical_sha256(ast.dump(constructor, include_attributes=False))

    payload = project_config_payload(config)
    # A section's earliest consumer owns it; downstream reuse is guarded by
    # upstream receipt hashes. Unknown sections conservatively belong to S0.
    owners = {
        "locate_anything": "S2", "qwen": "S5", "qwen_candidates": "S5",
        "evidence": "S4", "triggering": "S4", "open_vocabulary": "S4",
        "fusion": "S6", "tracking": "S3", "video": "S0",
        "outputs": "S7", "render": "S7", "encoding": "S7",
        "research_budget": "S4", "localization": "S2", "yolo": "S2",
    }
    configs: dict[str, dict[str, Any]] = {f"S{i}": {} for i in range(8)}
    for key, value in payload.items():
        if key == "video" and isinstance(value, dict):
            video_owners = {
                "shared_screenshot_scan": "S1", "screenshot_cache_bytes": "S1",
                "preview_decode_strategy": "S0", "evidence_decode_strategy": "S4",
                "encoder_backend": "S7", "ffmpeg_path": "S7",
            }
            for option, setting in value.items():
                configs[video_owners.get(option, "S0")].setdefault(key, {})[option] = setting
        elif key == "performance" and isinstance(value, dict):
            for option, setting in value.items():
                owner = "S4" if option in {"preprocess_workers", "preprocess_queue_size"} else "S2"
                configs[owner].setdefault(key, {})[option] = setting
        elif key == "locate_first" and isinstance(value, dict):
            association = {"max_gap_seconds", "minimum_iou", "maximum_center_distance", "minimum_match_score", "minimum_match_margin"}
            for option, setting in value.items():
                owner = "S3" if option in association else "S5" if option == "qwen_every_screenshot" else "S1"
                configs[owner].setdefault(key, {})[option] = setting
        else:
            configs[owners.get(key, "S0")][key] = value
    extra_seeds = {
        "S0": [("preflight", "run_preflight")],
        "S2": [("adapters.locate_anything", "LocateAnythingAdapter"), ("locate_service", "LocateServiceSession")],
        "S5": [("adapters.qwen", "Qwen35Adapter"), ("qwen_worker", "QwenProcessClient")],
    }
    result: dict[str, dict[str, str]] = {}
    for stage in configs:
        seeds = [("research_stages", f"ResearchStageExecutor._run_{stage.lower()}"),
                 ("research_stages", "ResearchStageExecutor.execute"),
                 ("stage_runtime", "StageLedger")]
        dependencies = closure(seeds + extra_seeds.get(stage, []))
        if stage == "S2":
            service_script = Path(config.project_root) / "scripts" / "locate_anything_service.py"
            if service_script.is_file():
                dependencies["scripts/locate_anything_service.py"] = sha256_file(service_script)
        result[stage] = {
            "code_sha256": canonical_sha256({"shared": shared, "symbols": dependencies}),
            "config_sha256": canonical_sha256(configs[stage]),
        }
    return result


__all__ = [
    "canonical_sha256",
    "project_config_digest",
    "project_config_payload",
    "python_source_digest",
    "sha256_file",
    "hash_file_cache",
    "uncached_hashes",
    "stage_dependency_fingerprints",
]
