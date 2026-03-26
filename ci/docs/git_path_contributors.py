#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show contributors and commit stats for a git path."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Repository-relative or absolute path to inspect",
    )
    parser.add_argument(
        "--changed-files-file",
        help="Path to a newline-delimited file containing changed files for reviewer selection.",
    )
    parser.add_argument(
        "--pr-author-login",
        help="GitHub login of the PR author to exclude from reviewer selection.",
    )
    return parser.parse_args()


def run_git_command(args, cwd):
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def find_repo_root():
    return Path(
        run_git_command(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()).strip()
    )


def normalize_target_path(repo_root, target_path):
    candidate = Path(target_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (repo_root / candidate).resolve()

    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside repository: {target_path}") from exc


def load_git_history(repo_root, relative_paths):
    if isinstance(relative_paths, str):
        relative_paths = [relative_paths]

    return run_git_command(
        [
            "git",
            "log",
            "--numstat",
            "--date=iso-strict",
            "--pretty=format:commit %H%nauthor %an%nauthor_email %ae%ntime %ai%n",
            "--",
            *relative_paths,
        ],
        cwd=repo_root,
    )


def safe_int(value):
    return int(value) if value.isdigit() else 0


def parse_commits(log_output):
    commits = []
    current = None

    for raw_line in log_output.splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            continue

        if line.startswith("commit "):
            if current is not None:
                commits.append(current)
            current = {
                "commit": line.split(" ", 1)[1],
                "author": "",
                "author_email": "",
                "time": "",
                "added_lines": 0,
                "deleted_lines": 0,
                "files": [],
            }
            continue

        if current is None:
            continue

        if line.startswith("author "):
            current["author"] = line.split(" ", 1)[1]
            continue

        if line.startswith("author_email "):
            current["author_email"] = line.split(" ", 1)[1]
            continue

        if line.startswith("time "):
            current["time"] = line.split(" ", 1)[1]
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            continue

        added, deleted, path = parts
        current["added_lines"] += safe_int(added)
        current["deleted_lines"] += safe_int(deleted)
        current["files"].append(path)

    if current is not None:
        commits.append(current)

    return commits


def group_commits_by_author(commits):
    grouped = defaultdict(list)
    for commit in commits:
        grouped[commit["author"]].append(commit)

    authors = []
    for author, author_commits in grouped.items():
        sorted_commits = sorted(author_commits, key=lambda item: item["time"], reverse=True)
        total_added = sum(item["added_lines"] for item in sorted_commits)
        total_deleted = sum(item["deleted_lines"] for item in sorted_commits)
        authors.append(
            {
                "author": author,
                "commit_count": len(sorted_commits),
                "total_added_lines": total_added,
                "total_deleted_lines": total_deleted,
                "total_lines": total_added + total_deleted,
                "commits": sorted_commits,
            }
        )

    return sorted(
        authors,
        key=lambda item: (item["total_lines"], item["commit_count"], item["author"]),
        reverse=True,
    )


def group_commits_by_file_and_author(target_files, commits):
    target_set = set(target_files)
    grouped = {path: defaultdict(list) for path in target_set}

    for commit in commits:
        for path in commit["files"]:
            if path not in target_set:
                continue
            grouped[path][(commit["author"], commit.get("author_email", ""))].append(commit)

    ranked = {}
    for path, authors in grouped.items():
        path_candidates = []
        for (author, author_email), author_commits in authors.items():
            sorted_commits = sorted(
                author_commits,
                key=lambda item: (item["time"], item["commit"]),
                reverse=True,
            )
            path_candidates.append(
                {
                    "author": author,
                    "author_email": author_email,
                    "commit_count": len(sorted_commits),
                    "commit": sorted_commits[0]["commit"],
                    "time": sorted_commits[0]["time"],
                }
            )

        ranked[path] = sorted(
            path_candidates,
            key=lambda item: (
                item["commit_count"],
                item["time"],
                item["author"],
                item["author_email"],
            ),
            reverse=True,
        )

    return ranked


def collect_top_reviewers(target_files, commits, pr_author_login, resolve_login):
    ranked_candidates = group_commits_by_file_and_author(target_files, commits)
    selected = []
    selected_logins = set()

    for path in target_files:
        for candidate in ranked_candidates.get(path, []):
            login = resolve_login(candidate)
            if not login or login == pr_author_login:
                continue
            if login in selected_logins:
                break

            selected.append(
                {
                    "login": login,
                    "source_file": path,
                    "author": candidate["author"],
                    "author_email": candidate["author_email"],
                    "commit": candidate["commit"],
                    "commit_count": candidate["commit_count"],
                }
            )
            selected_logins.add(login)
            break

    return selected


def render_reviewer_payload(reviewers):
    return json.dumps({"reviewers": reviewers}, indent=2, sort_keys=True)


def normalize_changed_files(repo_root, changed_files):
    normalized = []
    for raw_path in changed_files:
        path = raw_path.strip()
        if not path:
            continue
        normalized.append(normalize_target_path(repo_root, path))
    return normalized


def read_changed_files(file_path):
    return Path(file_path).read_text(encoding="utf-8").splitlines()


def github_api_request(url, token):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def infer_username_candidates(candidate):
    usernames = []
    email = candidate.get("author_email", "").strip()
    if email and "@" in email:
        local_part = email.split("@", 1)[0]
        local_part = local_part.split("+", 1)[0]
        usernames.append(local_part)

    author = candidate.get("author", "").strip()
    if author:
        usernames.append(author)
        usernames.append(author.replace(" ", ""))

    filtered = []
    seen = set()
    for username in usernames:
        normalized = username.strip()
        if not normalized:
            continue
        if not re.fullmatch(r"[A-Za-z0-9-]{1,39}", normalized):
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        filtered.append(normalized)
    return filtered


def make_github_login_resolver(repo_full_name, token):
    commit_cache = {}
    search_cache = {}

    def resolve_login(candidate):
        commit = candidate["commit"]
        if commit not in commit_cache:
            url = f"https://api.github.com/repos/{repo_full_name}/commits/{commit}"
            try:
                payload = github_api_request(url, token)
            except urllib.error.HTTPError:
                payload = {}
            commit_cache[commit] = (
                payload.get("author") or payload.get("committer") or {}
            ).get("login")

        if commit_cache[commit]:
            return commit_cache[commit]

        for username in infer_username_candidates(candidate):
            lowered = username.lower()
            if lowered not in search_cache:
                query = urllib.parse.quote(f"{username} in:login")
                url = f"https://api.github.com/search/users?q={query}&per_page=5"
                try:
                    payload = github_api_request(url, token)
                except urllib.error.HTTPError:
                    payload = {}
                exact_match = None
                for item in payload.get("items", []):
                    login = item.get("login", "")
                    if login.lower() == lowered:
                        exact_match = login
                        break
                search_cache[lowered] = exact_match

            if search_cache[lowered]:
                return search_cache[lowered]

        return None

    return resolve_login


def render_report(target_path, authors):
    lines = [f"Path: {target_path}", f"Authors: {len(authors)}", ""]

    for author in authors:
        lines.append(
            "Author: {author} | Commits: {commit_count} | Lines: +{added} -{deleted} | Total: {total}".format(
                author=author["author"],
                commit_count=author["commit_count"],
                added=author["total_added_lines"],
                deleted=author["total_deleted_lines"],
                total=author["total_lines"],
            )
        )
        for commit in author["commits"]:
            lines.append(
                "  Commit: {commit} | Time: {time} | Lines: +{added} -{deleted} | Files: {files}".format(
                    commit=commit["commit"],
                    time=commit["time"],
                    added=commit["added_lines"],
                    deleted=commit["deleted_lines"],
                    files=", ".join(commit["files"]),
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip()


def main():
    args = parse_args()

    try:
        repo_root = find_repo_root()
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.changed_files_file:
        if not args.pr_author_login:
            print("--pr-author-login is required with --changed-files-file", file=sys.stderr)
            return 1

        github_token = os.getenv("GITHUB_TOKEN")
        github_repo = os.getenv("GITHUB_REPOSITORY")
        if not github_token or not github_repo:
            print(
                "GITHUB_TOKEN and GITHUB_REPOSITORY are required for reviewer selection",
                file=sys.stderr,
            )
            return 1

        try:
            changed_files = normalize_changed_files(
                repo_root, read_changed_files(args.changed_files_file)
            )
            if not changed_files:
                print(render_reviewer_payload([]))
                return 0

            log_output = load_git_history(repo_root, changed_files)
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            message = getattr(exc, "stderr", "") or str(exc)
            print(str(message).strip(), file=sys.stderr)
            return 1

        commits = parse_commits(log_output)
        resolver = make_github_login_resolver(github_repo, github_token)
        reviewers = collect_top_reviewers(
            changed_files,
            commits,
            pr_author_login=args.pr_author_login,
            resolve_login=resolver,
        )
        print(render_reviewer_payload(reviewers))
        return 0

    if not args.path:
        print("path is required unless --changed-files-file is used", file=sys.stderr)
        return 1

    try:
        relative_path = normalize_target_path(repo_root, args.path)
        log_output = load_git_history(repo_root, relative_path)
    except (subprocess.CalledProcessError, ValueError) as exc:
        message = getattr(exc, "stderr", "") or str(exc)
        print(str(message).strip(), file=sys.stderr)
        return 1

    commits = parse_commits(log_output)
    authors = group_commits_by_author(commits)
    print(render_report(relative_path, authors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
