from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGES_WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"
DECISIONS = ROOT / "docs" / "DECISIONS.md"


def test_pages_deploy_action_uses_a_narrow_upstream_warning_suppression() -> None:
    workflow = PAGES_WORKFLOW.read_text()
    decisions = DECISIONS.read_text()

    assert "uses: actions/deploy-pages@v5" in workflow
    assert "NODE_OPTIONS: --disable-warning=DEP0040" in workflow
    assert "https://github.com/actions/deploy-pages/issues/413" in workflow
    assert "NODE_NO_WARNINGS" not in workflow
    assert "--no-warnings" not in workflow
    assert "--no-deprecation" not in workflow
    assert "https://github.com/actions/deploy-pages/issues/413" in decisions
    assert "process warnings." in decisions
