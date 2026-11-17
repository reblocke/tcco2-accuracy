"""Prepare a reviewed two-Mac Word import from the current remote main."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from check_package import CONTRACT, inspect_word, sha, strip_empty_pandoc_parts
from sources import ROLES

ROOT = Path(__file__).resolve().parents[2]
WIP = "works-in-progress"
SSH = [
    "ssh",
    "-4",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "reblocke@brians-mac-mini.local",
]


def choose(old, current):
    changes = {}
    for role in ROLES:
        known = {digest for host in current for kind, _, digest in old[host] if kind == role}
        new = set()
        for host, rows in current.items():
            for kind, name, digest in rows:
                if kind != role or digest == old.get("imported", {}).get(role):
                    continue
                previous = {h for k, n, h in old[host] if k == role and n == name}
                if (previous and digest not in previous) or (not previous and digest not in known):
                    new.add(digest)
        if len(new) > 1:
            versions = [
                (host, name, h[:12])
                for host, rows in current.items()
                for k, name, h in rows
                if k == role and h in new
            ]
            raise ValueError(f"Compare divergent {role} revisions in Word: {versions}")
        if new:
            changes[role] = new.pop()
    return changes


def read(host, digest=None):
    script = Path(__file__).with_name("sources.py")
    args = (
        [sys.executable, str(script), str(ROOT / "Drafts")]
        if host == "Air"
        else [*SSH, "python3", "-", "Research/tcco2-accuracy/Drafts"]
    )
    if digest:
        args.append(digest)
    try:
        return json.loads(
            subprocess.check_output(
                args, timeout=180, input=script.read_bytes() if host == "mini" else None
            )
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise SystemExit(f"{host} source unchecked. Restore access and rerun.") from exc


def cmd(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-reviewed", metavar="DIGEST")
    args = parser.parse_args()
    cmd("git", "fetch", "origin", "main")
    base = cmd("git", "rev-parse", "origin/main")
    name = f"{WIP}/source-revisions.json"
    initial = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{base}:{name}"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).returncode
        != 0
    )
    old = json.loads((ROOT / name).read_text() if initial else cmd("git", "show", f"{base}:{name}"))
    current = {host: read(host) for host in ("Air", "mini")}
    changes = choose(old, current)
    if not changes:
        print("Both Macs checked. No new Word revisions.")
        return
    if initial:
        raise SystemExit(
            "Initial manuscript PR is open. Compare new edits to this package until it reaches main."
        )
    approved_figures = {sha(p.read_bytes()): p for p in (ROOT / WIP).glob("*.png")}
    snapshot = {}
    for role, h in changes.items():
        host = next(host for host, rows in current.items() if any(row[2] == h for row in rows))
        snapshot[role] = read(host, h)
    signature = hashlib.sha256(
        json.dumps([base, current, snapshot], sort_keys=True).encode()
    ).hexdigest()
    review = ROOT / ".pytest_tmp" / "manuscript-import" / signature
    review.mkdir(parents=True, exist_ok=True)
    # Produce inert HTML for conversion. Source text cannot introduce a link or image.
    import html

    for role, items in snapshot.items():
        parts = []
        for item in items:
            if isinstance(item, str):
                parts.append("<p>" + html.escape(item) + "</p>")
            elif isinstance(item, dict):
                figure = approved_figures.get(item["image"])
                parts.append(
                    '<img width="540" src="' + html.escape(str(figure), quote=True) + '">'
                    if figure
                    else "<p>[Source illustration requires manuscript review.]</p>"
                )
            else:
                parts.append(
                    "<table>"
                    + "".join(
                        "<tr>" + "".join("<td>" + html.escape(s) + "</td>" for s in row) + "</tr>"
                        for row in item
                    )
                    + "</table>"
                )
        source = review / (role + ".html")
        source.write_text("\n".join(parts))
        dest = review / ROLES[role]
        if not dest.exists():
            cmd("pandoc", str(source), "-o", str(dest))
            strip_empty_pandoc_parts(dest)
    # Bind approval to the exact reviewed bytes and the base/source revisions.
    files = [review / ROLES[role] for role in sorted(changes)]
    token = hashlib.sha256(
        signature.encode() + b"".join(hashlib.sha256(f.read_bytes()).digest() for f in files)
    ).hexdigest()
    if args.publish_reviewed != token:
        print(
            f"Review visible text and tables against current main in {review}. Only approved manuscript figures are carried forward. After review run make manuscript-wip WIP_ARGS='--publish-reviewed {token}'."
        )
        return
    branch = "manuscript/import-" + token[:16]
    with tempfile.TemporaryDirectory() as tmp:
        wt = Path(tmp) / "tree"
        cmd("git", "worktree", "add", "--detach", str(wt), base)
        try:
            contract = json.loads((wt / CONTRACT).read_text())
            reviewed = contract["manuscript_wip"]["reviewed"]
            figures = {sha((wt / path).read_bytes()) for path in reviewed if path.endswith(".png")}
            for f in files:
                data = f.read_bytes()
                inspect_word(data, figures)
                path = f"{WIP}/{f.name}"
                (wt / path).write_bytes(data)
                if sha(data) not in reviewed[path]:
                    reviewed[path].append(sha(data))
            (wt / CONTRACT).write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
            current["imported"] = {**old.get("imported", {}), **changes}
            (wt / name).write_text(json.dumps(current, indent=2) + "\n")
            cmd(sys.executable, "scripts/manuscript/check_package.py", cwd=wt)
            cmd("git", "add", "--", WIP, str(CONTRACT), cwd=wt)
            cmd("git", "commit", "-m", "Update reviewed manuscript drafts", cwd=wt)
            try:
                cmd("git", "push", "origin", "HEAD:refs/heads/" + branch, cwd=wt)
            except subprocess.CalledProcessError:
                cmd("git", "fetch", "origin", branch, cwd=wt)
                if cmd("git", "rev-parse", "FETCH_HEAD^{tree}", cwd=wt) != cmd(
                    "git", "rev-parse", "HEAD^{tree}", cwd=wt
                ):
                    raise
            existing = cmd(
                "gh",
                "pr",
                "list",
                "--head",
                branch,
                "--base",
                "main",
                "--json",
                "url",
                "--jq",
                ".[0].url // empty",
                cwd=wt,
            )
            if existing:
                print(existing)
                return
            print(
                cmd(
                    "gh",
                    "pr",
                    "create",
                    "--draft",
                    "--base",
                    "main",
                    "--head",
                    branch,
                    "--title",
                    "Update reviewed manuscript drafts",
                    "--body",
                    "Reviewed public Word copies from the two-Mac manuscript inventory.",
                    cwd=wt,
                )
            )
        finally:
            cmd("git", "worktree", "remove", "--force", str(wt))


if __name__ == "__main__":
    os.umask(0o077)
    main()
