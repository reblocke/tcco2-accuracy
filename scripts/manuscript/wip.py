"""Prepare a reviewed two-Mac Word import from the current remote main."""

import argparse
import hashlib
import html
import json
import os
import re
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


def choose(old, current, public_hashes):
    changes = {}
    for role in ROLES:
        accepted = {old.get("imported", {}).get(role), public_hashes.get(role)} - {None}
        known = {h for host in current for k, _, h in old[host] if k == role}
        new = set()
        from_current = set()
        for host, rows in current.items():
            previous = [(n, h) for k, n, h in old[host] if k == role]
            for kind, name, digest in rows:
                if kind != role or digest in accepted:
                    continue
                same_name = {h for n, h in previous if n == name}
                if digest in same_name or (not same_name and digest in known):
                    continue
                new.add(digest)
                # A different filename on the same Mac says nothing about this file's base.
                if same_name and same_name <= accepted:
                    from_current.add(digest)
        versions = [
            (host, name, h[:12])
            for host, rows in current.items()
            for k, name, h in rows
            if k == role and h in new
        ]
        if len(new) > 1:
            raise ValueError(f"Compare divergent {role} revisions in Word: {versions}")
        if new and not from_current:
            raise ValueError(
                f"Unverified predecessor for {role} (stale, renamed or ambiguous copy): {versions}. Compare with current main in Word, save the reconciliation under the current source filename on its Mac, and archive superseded copies outside the import folders."
            )
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
    public_hashes = {
        role: sha(
            (ROOT / WIP / file).read_bytes()
            if initial
            else subprocess.check_output(["git", "show", f"{base}:{WIP}/{file}"], cwd=ROOT)
        )
        for role, file in ROLES.items()
    }
    changes = choose(old, current, public_hashes)
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
    inventory = {**current, "imported": {**old.get("imported", {}), **changes}}
    inventory_bytes = (json.dumps(inventory, indent=2) + "\n").encode()
    (review / "source-revisions.json").write_bytes(inventory_bytes)
    # Produce inert HTML for conversion. Source text cannot introduce a link or image.
    for role, source_document in snapshot.items():
        items = source_document["blocks"]
        equations = source_document["equations"]

        def visible(text):
            return re.sub(
                r"OMML[0-9a-f]{64}",
                lambda m: "<code>" + m[0] + "</code>" if m[0] in equations else m[0],
                html.escape(text),
            )

        parts = []
        for item in items:
            if isinstance(item, str):
                parts.append("<p>" + visible(item) + "</p>")
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
                        "<tr>" + "".join("<td>" + visible(s) + "</td>" for s in row) + "</tr>"
                        for row in item
                    )
                    + "</table>"
                )
        source = review / (role + ".html")
        source.write_text("\n".join(parts))
        dest = review / ROLES[role]
        if not dest.exists():
            if equations:

                def restore_math(node):
                    if node.get("t") == "Code" and node["c"][1] in equations:
                        return {"t": "RawInline", "c": ["openxml", equations[node["c"][1]]]}
                    return node

                document = json.loads(
                    cmd("pandoc", str(source), "-t", "json"), object_hook=restore_math
                )
                subprocess.run(
                    ["pandoc", "-f", "json", "-o", str(dest)],
                    input=json.dumps(document).encode(),
                    check=True,
                )
            else:
                cmd("pandoc", str(source), "-o", str(dest))
            strip_empty_pandoc_parts(dest)
    # Bind approval to the exact reviewed bytes and the base/source revisions.
    files = [review / ROLES[role] for role in sorted(changes)]
    token = hashlib.sha256(
        signature.encode() + b"".join(hashlib.sha256(f.read_bytes()).digest() for f in files)
    ).hexdigest()
    if args.publish_reviewed != token:
        print(
            f"Review visible text, tables and source-revisions.json against current main in {review}. Only approved manuscript figures are carried forward. After review run make manuscript-wip WIP_ARGS='--publish-reviewed {token}'."
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
            (wt / name).write_bytes(inventory_bytes)
            inventory_hash = sha((wt / name).read_bytes())
            if inventory_hash not in reviewed[name]:
                reviewed[name].append(inventory_hash)
            (wt / CONTRACT).write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
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
