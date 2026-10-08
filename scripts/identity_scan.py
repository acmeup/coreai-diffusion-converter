#!/usr/bin/env python3
# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Identity and personal-information scan for this repository.

Checks the working tree, commit history and commit messages for:

* tokens listed in ``.git/identity-tokens`` (one per line, case-insensitive). The list lives
  inside ``.git`` so it is never committed; a missing list is an error, so the check cannot pass
  silently;
* paths under the macOS user-home and external-volume roots;
* any e-mail address other than the project's no-reply address;
* git trailer lines of the ``<Word>-By:`` form.

No pattern literal appears in this file: each one is assembled from fragments at run time, so the
scanner passes over itself.

    python scripts/identity_scan.py --tree --commits HEAD
    python scripts/identity_scan.py --message .git/COMMIT_EDITMSG
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_FRAGMENTS = (("/", "Us", "ers", "/"), ("/", "Vol", "umes", "/"))
TRAILER_TAIL = "-" + "B" + "y" + ":"
ALLOWED_EMAIL = "".join(("221633905", "+", "acme", "up", "@", "users", ".", "noreply", ".", "git", "hub", ".", "com"))
_LOCAL = "[A-Za-z0-9._%+" + "-]+"
_DOMAIN = "[A-Za-z0-9.-]+" + r"\." + "[A-Za-z]{2,}"
EMAIL_RE = re.compile(_LOCAL + "@" + _DOMAIN)
TRAILER_RE = re.compile(r"(?im)^[ \t]*[A-Za-z]+(?:-[A-Za-z]+)*" + re.escape(TRAILER_TAIL))
PATH_RES = [re.compile(re.escape("".join(f)), re.I) for f in ROOT_FRAGMENTS]


@dataclass(frozen=True)
class Finding:
    where: str
    what: str

    def __str__(self) -> str:
        return f"{self.where}: {self.what}"


def git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=check)


def repo_root(cwd: Path | None = None) -> Path:
    return Path(git("rev-parse", "--show-toplevel", cwd=cwd).stdout.decode().strip())


def load_tokens(cwd: Path | None = None, tokens_file: Path | None = None) -> list[str]:
    if tokens_file is None:
        git_dir = git("rev-parse", "--absolute-git-dir", cwd=cwd).stdout.decode().strip()
        tokens_file = Path(git_dir) / "identity-tokens"
    if not tokens_file.is_file():
        raise SystemExit(f"identity scan: {tokens_file.name} is missing from the git directory; "
                         "create it (one forbidden token per line) before scanning")
    tokens = [t.strip().lower() for t in tokens_file.read_text(encoding="utf-8").splitlines()]
    tokens = [t for t in tokens if t and not t.startswith("#")]
    if not tokens:
        raise SystemExit("identity scan: the token list is empty")
    return tokens


def scan_text(text: str, where: str, tokens: list[str], *, binary: bool = False) -> list[Finding]:
    found: list[Finding] = []
    lower = text.lower()
    for t in tokens:
        if t in lower:
            found.append(Finding(where, f"forbidden token #{tokens.index(t) + 1}"))
    for rx in PATH_RES:
        if rx.search(text):
            found.append(Finding(where, "machine path"))
    if binary:
        return found
    for m in EMAIL_RE.finditer(text):
        if m.group(0).lower() != ALLOWED_EMAIL:
            found.append(Finding(where, "e-mail address other than the project no-reply address"))
    if TRAILER_RE.search(text):
        found.append(Finding(where, "git trailer line"))
    return found


def _decode(data: bytes) -> tuple[str, bool]:
    binary = b"\0" in data[:8192]
    return data.decode("utf-8", errors="replace"), binary


def tree_files(root: Path) -> list[str]:
    """Tracked, staged and untracked-but-not-ignored files (the latter so an unstaged working tree
    can be checked before its first commit)."""
    out = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", cwd=root).stdout
    return sorted({p for p in out.decode().split("\0") if p})


def scan_tree(root: Path, tokens: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for rel in tree_files(root):
        path = root / rel
        if not path.is_file() or path.is_symlink():
            continue
        text, binary = _decode(path.read_bytes())
        findings += scan_text(rel, f"{rel} (path)", tokens, binary=True)
        findings += scan_text(text, rel, tokens, binary=binary)
    return findings


def scan_commits(root: Path, rev_range: str, tokens: list[str]) -> list[Finding]:
    if rev_range == "HEAD" and git("rev-parse", "--verify", "-q", "HEAD", cwd=root, check=False).returncode:
        print("identity scan: no commits yet; history check skipped")
        return []
    findings: list[Finding] = []
    shas = git("rev-list", rev_range, cwd=root).stdout.decode().split()
    for sha in shas:
        meta = git("log", "-1", "--format=%an%n%ae%n%cn%n%ce%n%B", sha, cwd=root).stdout.decode("utf-8", "replace")
        an, ae, cn, ce, *body = meta.split("\n")
        findings += scan_text("\n".join(body), f"commit {sha[:12]} message", tokens)
        for label, value in (("author name", an), ("committer name", cn)):
            if value != "AcmeUp Inc.":
                findings.append(Finding(f"commit {sha[:12]}", f"{label} is not AcmeUp Inc."))
        for label, value in (("author e-mail", ae), ("committer e-mail", ce)):
            if value.lower() != ALLOWED_EMAIL:
                findings.append(Finding(f"commit {sha[:12]}", f"{label} is not the project no-reply address"))
        for rel in git("ls-tree", "-r", "-z", "--name-only", sha, cwd=root).stdout.decode().split("\0"):
            if not rel:
                continue
            blob = git("show", f"{sha}:{rel}", cwd=root, check=False).stdout
            text, binary = _decode(blob)
            findings += scan_text(rel, f"commit {sha[:12]} {rel} (path)", tokens, binary=True)
            findings += scan_text(text, f"commit {sha[:12]} {rel}", tokens, binary=binary)
    return findings


def scan_message(path: Path, tokens: list[str]) -> list[Finding]:
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if not ln.startswith("#")]
    return scan_text("\n".join(lines), "commit message", tokens)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", action="store_true", help="scan the working tree (default)")
    ap.add_argument("--commits", metavar="RANGE", help="scan commits in RANGE (metadata and contents)")
    ap.add_argument("--message", type=Path, metavar="FILE", help="scan a commit message file")
    ap.add_argument("--tokens-file", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = repo_root()
    tokens = load_tokens(root, args.tokens_file)
    findings: list[Finding] = []
    if args.message:
        findings += scan_message(args.message, tokens)
    if args.commits:
        findings += scan_commits(root, args.commits, tokens)
    if args.tree or not (args.message or args.commits):
        findings += scan_tree(root, tokens)
    for f in findings:
        print(f"identity scan: {f}", file=sys.stderr)
    if findings:
        print(f"identity scan: FAILED ({len(findings)} finding(s))", file=sys.stderr)
        return 1
    print("identity scan: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
