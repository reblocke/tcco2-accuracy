"""Two Macs and a local bare remote exercise the manuscript import without public writes."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not shutil.which("pandoc"), reason="Word import uses Pandoc on the Air")
def test_two_mac_review_and_publish(tmp_path):
    home = tmp_path / "home"
    mini = tmp_path / "mini"
    checkout = tmp_path / "checkout"
    remote = tmp_path / "origin.git"
    tools = tmp_path / "bin"
    tools.mkdir()
    env = {
        **os.environ,
        "HOME": str(home),
        "MOCK_MINI": str(mini),
        "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }

    def command(*args, ok=True):
        return subprocess.run(args, cwd=checkout, env=env, text=True, capture_output=True, check=ok)

    checkout.mkdir()
    home.mkdir()
    repo = checkout / "works-in-progress"
    shutil.copytree(ROOT / "works-in-progress", repo)
    shutil.copytree(
        ROOT / "scripts/manuscript",
        checkout / "scripts/manuscript",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (checkout / "docs").mkdir()
    shutil.copy(ROOT / "docs/data_release_contract.json", checkout / "docs")
    roots = [checkout / "Drafts", mini / "Research/tcco2-accuracy/Drafts"]
    for p in [
        *roots,
        *[h / "Documents/Codex/llm-readiness-working/tcco2-accuracy/Drafts" for h in (home, mini)],
    ]:
        p.mkdir(parents=True)

    def revisions():
        digest = hashlib.sha256((repo / "letter.docx").read_bytes()).hexdigest()
        row = [["letter", "Letter.docx", digest]]
        return {"Air": row, "mini": row, "imported": {"letter": digest}}

    for root in roots:
        shutil.copy(repo / "letter.docx", root / "Letter.docx")
    (repo / "source-revisions.json").write_text(json.dumps(revisions()))
    for name, script in [
        ("ssh", 'export HOME="$MOCK_MINI"\nwhile [ "$1" != python3 ]; do shift; done\nexec "$@"'),
        ("gh", 'if [ "$2" = create ]; then echo reviewed-pr; fi'),
    ]:
        path = tools / name
        path.write_text("#!/bin/sh\n" + script + "\n")
        path.chmod(0o755)
    command("git", "init", "-b", "main")
    command("git", "add", "docs", "works-in-progress", "scripts")
    command("git", "commit", "-m", "Initial fixture")
    command("git", "init", "--bare", str(remote))
    command("git", "remote", "add", "origin", str(remote))
    command("git", "push", "origin", "main")

    def run(*args, ok=True):
        return command(sys.executable, "scripts/manuscript/wip.py", *args, ok=ok)

    assert "No new" in run().stdout
    (roots[0] / "~$Letter.docx").write_bytes(b"lock")
    assert "No new" in run().stdout

    def revise(path, new):
        with ZipFile(repo / "letter.docx") as src, ZipFile(path, "w") as dst:
            for name in src.namelist():
                content = src.read(name)
                if name == "word/document.xml":
                    content = content.replace(b"Research letter in preparation", new)
                dst.writestr(name, content)
            dst.writestr("word/embeddings/private.bin", b"private source object")

    revise(roots[1] / "Letter.docx", b"Revised letter for review")
    preview = run().stdout
    token = preview.split("--publish-reviewed ")[-1].split("'")[0]
    assert len(token) == 64
    revised = next((checkout / ".pytest_tmp").rglob("letter.docx"))
    with ZipFile(revised) as doc:
        assert b"Revised letter for review" in doc.read("word/document.xml")
        assert not any("private" in name or "comments" in name for name in doc.namelist())
        assert len([n for n in doc.namelist() if n.startswith("word/media/")]) == 2
    # Only the reviewed bytes can be published.
    good = revised.read_bytes()
    with ZipFile(revised, "a") as z:
        z.writestr("word/embeddings/private.bin", b"x")
    assert "Review visible" in run("--publish-reviewed", token).stdout
    bad_token = run().stdout.split("--publish-reviewed ")[-1].split("'")[0]
    assert "Unreviewed Word part" in run("--publish-reviewed", bad_token, ok=False).stderr
    revised.write_bytes(good)
    assert "reviewed-pr" in run("--publish-reviewed", token).stdout
    assert len(command("git", "ls-remote", "--heads", "origin").stdout.splitlines()) == 2
    revise(roots[0] / "Letter.docx", b"Independently revised on Air")
    assert "divergent" in run(ok=False).stderr
    shutil.copy(roots[1] / "Letter.docx", roots[0] / "Letter.docx")
    assert "Review visible" in run().stdout
    roots[1].rename(mini / "unavailable")
    assert "mini source unchecked" in run(ok=False).stderr
