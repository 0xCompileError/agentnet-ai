from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GUIDES = {
    "quickstart.md": (
        "# Quickstart",
        "agentnet init",
        "FakeLLM",
        "descriptor-only `.agentnet` artifact",
    ),
    "architecture.md": (
        "# Architecture Guide",
        "ReActAgent",
        "Graph containers",
        "Dependency injection",
    ),
    "training.md": (
        "# Training Guide",
        "Trainer",
        "Dataset",
        "Objective",
    ),
    "topology-search.md": (
        "# Topology Search Guide",
        "TopologySearchSpace",
        "TopologyOptimizer.search",
        "max_trials",
    ),
    "distributed-runtime.md": (
        "# Distributed Runtime Guide",
        "Scheduler",
        "ThreadPoolScheduler",
        "injected clients",
    ),
    "mcp.md": (
        "# MCP Guide",
        "MCPRegistry",
        "MCPToolAdapter",
        "allowlists",
    ),
    "langsmith.md": (
        "# LangSmith Guide",
        "LangSmithExporter",
        "injected client",
        "trace_from_context",
    ),
    "package-export.md": (
        "# Package Export Guide",
        "agentnet export",
        "export_package",
        "explicit dependency injection",
    ),
    "plugins.md": (
        "# Plugin Guide",
        "Milestone 19",
        "ConstraintPluginRegistry",
        "does not load executable code",
    ),
    "enterprise.md": (
        "# Enterprise Guide",
        "Do not serialize secrets",
        "approval",
        "descriptor-only",
    ),
}


def test_milestone_documentation_guides_exist_with_expected_content() -> None:
    for filename, phrases in GUIDES.items():
        guide = ROOT / "docs" / filename
        assert guide.is_file(), filename
        content = guide.read_text()

        for phrase in phrases:
            assert phrase in content, f"{filename} missing {phrase!r}"
        assert "```" in content, f"{filename} should include a concrete example"


def test_mkdocs_navigation_includes_milestone_guides() -> None:
    mkdocs = (ROOT / "mkdocs.yml").read_text()

    for filename in GUIDES:
        assert filename in mkdocs


def test_documentation_index_links_to_all_guides() -> None:
    index = (ROOT / "docs" / "index.md").read_text()

    assert "Milestone 17" in index
    for filename in GUIDES:
        assert f"]({filename})" in index
