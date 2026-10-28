from __future__ import annotations

from pathlib import Path
import shutil

from classroom_hybrid_ai.config import ProjectConfig
from classroom_hybrid_ai.research_backend import _code_fingerprint
from classroom_hybrid_ai.hashing import stage_dependency_fingerprints


def _config(project: Path) -> ProjectConfig:
    return ProjectConfig(
        source_path=project / "configs" / "default.yaml",
        project_root=project,
        repository_root=project.parent,
        payload={},
    )


def test_code_fingerprint_ignores_generated_egg_info_but_tracks_package_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    package = project / "src" / "classroom_hybrid_ai"
    package.mkdir(parents=True)
    (package / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "configs").mkdir()
    (project / "scripts").mkdir()

    config = _config(project)
    before = _code_fingerprint(config)

    egg_info = project / "src" / "classroom_hybrid_ai.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated once\n", encoding="utf-8")
    assert _code_fingerprint(config) == before

    (egg_info / "PKG-INFO").write_text("generated again\n", encoding="utf-8")
    assert _code_fingerprint(config) == before

    (package / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert _code_fingerprint(config) != before


def test_stage_dependencies_separate_qwen_render_and_shared_changes(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    package = project / "src" / "classroom_hybrid_ai"
    shutil.copytree(source / "src" / "classroom_hybrid_ai", package, ignore=shutil.ignore_patterns("__pycache__"))
    config = _config(project)
    before = stage_dependency_fingerprints(config)
    (project / "README.md").write_text("documentation changed", encoding="utf-8")
    assert stage_dependency_fingerprints(config) == before
    qwen = package / "adapters" / "qwen.py"
    original_qwen = qwen.read_text(encoding="utf-8")
    qwen.write_text(original_qwen.replace('class Qwen35Adapter:', 'class Qwen35Adapter:\n    stage_fingerprint_probe = 1'), encoding="utf-8")
    after = stage_dependency_fingerprints(config)
    assert [stage for stage in before if before[stage] != after[stage]] == ["S5"]
    qwen.write_text(original_qwen, encoding="utf-8")
    output = package / "outputs_v3.py"
    original_output = output.read_text(encoding="utf-8")
    # Change the AST body of the rendering implementation regardless of its
    # wrapper shape, keeping this test independent of signature formatting.
    import ast
    tree = ast.parse(original_output)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_render_videos_once_impl')
    lines = original_output.splitlines(keepends=True)
    lines.insert(node.body[0].lineno - 1, '    render_fingerprint_probe = 1\n')
    output.write_text(''.join(lines), encoding='utf-8')
    after = stage_dependency_fingerprints(config)
    assert [stage for stage in before if before[stage] != after[stage]] == ["S7"]
    output.write_text(original_output, encoding="utf-8")
    shared = package / "storage.py"
    shared.write_text(shared.read_text(encoding="utf-8") + '\nFINGERPRINT_PROBE = 1\n', encoding='utf-8')
    after = stage_dependency_fingerprints(config)
    assert all(before[stage] != after[stage] for stage in before)


def test_stage_config_dependencies_are_separate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    before = stage_dependency_fingerprints(config)
    changed = ProjectConfig(config.source_path, config.project_root, config.repository_root, {"qwen": {"prompt_version": "compact_v1"}})
    after = stage_dependency_fingerprints(changed)
    assert [stage for stage in before if before[stage] != after[stage]] == ["S5"]
    changed = ProjectConfig(config.source_path, config.project_root, config.repository_root, {"video": {"encoder_backend": "nvenc"}})
    after = stage_dependency_fingerprints(changed)
    assert [stage for stage in before if before[stage] != after[stage]] == ["S7"]
