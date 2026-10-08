# Copyright 2026 AcmeUp Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bad samples are assembled from fragments, like the scanner's own patterns, so this file
passes the scan too."""

import importlib.util
import os
import subprocess
import sys

import pytest

from conftest import REPO

SCRIPT = REPO / "scripts" / "identity_scan.py"
spec = importlib.util.spec_from_file_location("identity_scan", SCRIPT)
scan = importlib.util.module_from_spec(spec)
sys.modules["identity_scan"] = scan
spec.loader.exec_module(scan)

TOKEN = "zz" + "secret" + "name"
HOME_PATH = "/" + "Us" + "ers" + "/someone/project"
VOLUME_PATH = "/" + "Vol" + "umes" + "/disk/x"
FOREIGN_EMAIL = "person" + "@" + "example" + ".org"
TRAILER = "Signed" + "-off" + "-" + "By" + ": somebody"
NOREPLY = scan.ALLOWED_EMAIL


@pytest.mark.parametrize("text", [f"hello {TOKEN} world", HOME_PATH, VOLUME_PATH, FOREIGN_EMAIL,
                                  f"subject\n\n{TRAILER}\n"])
def test_bad_samples_fail(text):
    assert scan.scan_text(text, "x", [TOKEN])


def test_noreply_and_plain_text_pass():
    assert scan.scan_text(f"Author: AcmeUp Inc. <{NOREPLY}>\nOrganization AcmeUp Inc.", "x", [TOKEN]) == []


def test_token_match_is_case_insensitive():
    assert scan.scan_text(TOKEN.upper(), "x", [TOKEN])


def run_git(repo, *args, env=None):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)


def make_repo(tmp_path, tokens=True):
    repo = tmp_path / "r"
    repo.mkdir()
    run_git(repo, "init", "-q", "-b", "main")
    run_git(repo, "config", "user.name", "AcmeUp Inc.")
    run_git(repo, "config", "user.email", NOREPLY)
    run_git(repo, "config", "commit.template", "")
    run_git(repo, "config", "core.hooksPath", ".no-hooks")
    if tokens:
        (repo / ".git" / "identity-tokens").write_text(TOKEN + "\n")
    return repo


def run_scan(repo, *args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=repo, capture_output=True, text=True)


def test_missing_tokens_file_fails(tmp_path):
    repo = make_repo(tmp_path, tokens=False)
    assert run_scan(repo, "--tree").returncode != 0


def test_tree_mode_finds_untracked_and_tracked(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "ok.txt").write_text("fine\n")
    assert run_scan(repo, "--tree").returncode == 0
    (repo / "bad.txt").write_text(HOME_PATH)
    r = run_scan(repo, "--tree")
    assert r.returncode == 1 and "bad.txt" in r.stderr


def test_unborn_head_is_handled(tmp_path):
    repo = make_repo(tmp_path)
    r = run_scan(repo, "--commits", "HEAD")
    assert r.returncode == 0 and "no commits" in r.stdout


def test_commit_message_and_author(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "a.txt").write_text("a\n")
    run_git(repo, "add", "a.txt")
    run_git(repo, "commit", "-q", "-m", "Clean message")
    assert run_scan(repo, "--commits", "HEAD").returncode == 0
    (repo / "b.txt").write_text("b\n")
    run_git(repo, "add", "b.txt")
    run_git(repo, "commit", "-q", "-m", f"Add b\n\n{TRAILER}")
    r = run_scan(repo, "--commits", "HEAD")
    assert r.returncode == 1 and "trailer" in r.stderr
    env = {**os.environ, "GIT_AUTHOR_NAME": "Some Person", "GIT_AUTHOR_EMAIL": FOREIGN_EMAIL}
    (repo / "c.txt").write_text("c\n")
    run_git(repo, "add", "c.txt")
    run_git(repo, "commit", "-q", "-m", "Add c", env=env)
    r = run_scan(repo, "--commits", "HEAD~1..HEAD")
    assert r.returncode == 1 and "author" in r.stderr


def test_message_mode(tmp_path):
    repo = make_repo(tmp_path)
    msg = tmp_path / "msg"
    msg.write_text(f"Subject\n# {TOKEN} in a comment line is ignored\n")
    assert run_scan(repo, "--message", str(msg)).returncode == 0
    msg.write_text(f"Subject mentioning {TOKEN}\n")
    assert run_scan(repo, "--message", str(msg)).returncode == 1


def test_scanner_passes_over_this_repository():
    if not (REPO / ".git" / "identity-tokens").is_file():
        pytest.skip("the token list is created per clone in .git/identity-tokens")
    r = run_scan(REPO, "--tree")
    assert r.returncode == 0, r.stderr


def test_scanner_source_contains_no_pattern_literal():
    src = SCRIPT.read_text()
    tokens = scan.load_tokens(REPO) if (REPO / ".git" / "identity-tokens").is_file() else []
    assert scan.scan_text(src, "scanner", tokens) == []
