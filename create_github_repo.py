"""Create the private CosCert repository using the existing Git Credential Manager.

Run from a normal terminal: python create_github_repo.py
The saved credential is held in memory only and sent only to api.github.com.
The script does not change proxy settings, repository visibility, or existing commits.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
OWNER = "EslamAhmedGaber"
NAME = "CosCert"
URL = f"https://github.com/{OWNER}/{NAME}"


def run(*args):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def main():
    git = shutil.which("git")
    if not git:
        raise RuntimeError("Git is required.")
    manager = shutil.which("git-credential-manager")
    if not manager:
        candidate = Path(git).parent.parent / "mingw64/bin/git-credential-manager.exe"
        if candidate.is_file():
            manager = str(candidate)
    if not manager:
        raise RuntimeError("Git Credential Manager was not found.")
    credential = subprocess.run(
        [manager, "get"],
        input=f"protocol=https\nhost=github.com\nusername={OWNER}\n\n",
        capture_output=True, text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
    )
    if credential.returncode:
        raise RuntimeError("The saved GitHub credential could not be retrieved.")
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if not token:
        raise RuntimeError("No saved GitHub credential was returned.")

    def api(path, data=None):
        request = urllib.request.Request(
            "https://api.github.com" + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Authorization": "Bearer " + token,
                     "Accept": "application/vnd.github+json",
                     "Content-Type": "application/json",
                     "User-Agent": "CosCert-repository-setup"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)

    profile = api("/user")
    if profile["login"].lower() != OWNER.lower():
        raise RuntimeError("The saved credential belongs to a different account; no changes made.")
    print("Authenticated as", profile["login"])
    try:
        repo = api(f"/repos/{OWNER}/{NAME}")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        repo = api("/user/repos", {
            "name": NAME,
            "description": "Reproducibility companion for CosCert: certified critical values and exact size for long-memory tests of the mean",
            "private": True,
            "auto_init": False,
        })
    if not repo.get("private"):
        raise RuntimeError("The repository is already public; stopped before uploading this private research package.")
    if repo["full_name"].lower() != f"{OWNER}/{NAME}".lower():
        raise RuntimeError("Unexpected repository identity.")

    if not (ROOT / ".git").exists():
        run(git, "init", "-b", "main")
    run(git, "config", "user.name", "Eslam Ahmed")
    run(git, "config", "user.email", "eslamahmed@eng.cu.edu.eg")
    # Repository-local helper avoids the shell dispatch used by the short alias.
    run(git, "config", "credential.helper", '"' + str(Path(manager)).replace("\\", "/") + '"')
    remotes = run(git, "remote").splitlines()
    if "origin" in remotes:
        if run(git, "remote", "get-url", "origin").rstrip("/").removesuffix(".git") != URL:
            raise RuntimeError("Existing origin differs; stopped without changing it.")
    else:
        run(git, "remote", "add", "origin", URL + ".git")
    run(git, "add", "--all")
    if run(git, "diff", "--cached", "--name-only"):
        run(git, "commit", "-m", "Add CosCert reproducibility package and verified release inputs")
    # Never force-push: unrelated remote history is rejected by Git.
    run(git, "push", "-u", "origin", "HEAD:main")
    local = run(git, "rev-parse", "HEAD")
    remote = api(f"/repos/{OWNER}/{NAME}/git/ref/heads/main")["object"]["sha"]
    if local != remote:
        raise RuntimeError("Remote verification failed: commit does not match.")
    receipt = {"url": URL, "private": True, "commit": remote}
    (ROOT / "repo_creation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Verified private repository:", URL)
    print("Commit:", remote)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Do not print subprocess output or headers: they may contain credentials.
        print("Repository setup stopped:", type(error).__name__, file=sys.stderr)
        if isinstance(error, RuntimeError):
            print(str(error), file=sys.stderr)
        elif isinstance(error, urllib.error.HTTPError):
            print("GitHub HTTP status:", error.code, file=sys.stderr)
        elif isinstance(error, urllib.error.URLError):
            print("GitHub is unreachable under the current network/proxy settings.", file=sys.stderr)
        sys.exit(1)
