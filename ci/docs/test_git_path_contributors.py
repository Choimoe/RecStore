#!/usr/bin/env python3

import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("git_path_contributors.py")
SPEC = importlib.util.spec_from_file_location("git_path_contributors", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


SAMPLE_LOG = """commit abc123
author Alice
time 2024-01-02 03:04:05 +0000

10\t2\tsrc/a.py
3\t1\tsrc/b.py

commit def456
author Bob
time 2024-02-03 04:05:06 +0000

-\t-\tbinary/model.bin
4\t0\tsrc/a.py

"""


REVIEWER_COMMITS = [
    {
        "commit": "c1",
        "author": "Alice",
        "author_email": "alice@example.com",
        "time": "2024-01-04 00:00:00 +0000",
        "added_lines": 1,
        "deleted_lines": 0,
        "files": ["src/a.py"],
    },
    {
        "commit": "c2",
        "author": "Alice",
        "author_email": "alice@example.com",
        "time": "2024-01-03 00:00:00 +0000",
        "added_lines": 2,
        "deleted_lines": 1,
        "files": ["src/a.py"],
    },
    {
        "commit": "c3",
        "author": "Bob",
        "author_email": "bob@example.com",
        "time": "2024-01-02 00:00:00 +0000",
        "added_lines": 3,
        "deleted_lines": 0,
        "files": ["src/a.py", "src/b.py"],
    },
    {
        "commit": "c4",
        "author": "Carol",
        "author_email": "carol@example.com",
        "time": "2024-01-01 00:00:00 +0000",
        "added_lines": 4,
        "deleted_lines": 0,
        "files": ["src/b.py"],
    },
]


class GitPathContributorsTest(unittest.TestCase):
    def test_parse_commits_from_numstat_log(self):
        commits = MODULE.parse_commits(SAMPLE_LOG)

        self.assertEqual(2, len(commits))
        self.assertEqual("abc123", commits[0]["commit"])
        self.assertEqual("Alice", commits[0]["author"])
        self.assertEqual("2024-01-02 03:04:05 +0000", commits[0]["time"])
        self.assertEqual(13, commits[0]["added_lines"])
        self.assertEqual(3, commits[0]["deleted_lines"])
        self.assertEqual(["src/a.py", "src/b.py"], commits[0]["files"])

        self.assertEqual("def456", commits[1]["commit"])
        self.assertEqual(4, commits[1]["added_lines"])
        self.assertEqual(0, commits[1]["deleted_lines"])
        self.assertEqual(["binary/model.bin", "src/a.py"], commits[1]["files"])

    def test_group_commits_by_author_and_sort(self):
        commits = MODULE.parse_commits(SAMPLE_LOG)
        authors = MODULE.group_commits_by_author(commits)

        self.assertEqual(["Alice", "Bob"], [author["author"] for author in authors])
        self.assertEqual(16, authors[0]["total_lines"])
        self.assertEqual(4, authors[1]["total_lines"])
        self.assertEqual(["abc123"], [commit["commit"] for commit in authors[0]["commits"]])

    def test_render_report_contains_commit_details(self):
        commits = MODULE.parse_commits(SAMPLE_LOG)
        authors = MODULE.group_commits_by_author(commits)
        report = MODULE.render_report("src", authors)

        self.assertIn("Path: src", report)
        self.assertIn("Author: Alice", report)
        self.assertIn("Commit: abc123", report)
        self.assertIn("Time: 2024-01-02 03:04:05 +0000", report)
        self.assertIn("Lines: +13 -3", report)
        self.assertIn("Files: src/a.py, src/b.py", report)

    def test_collect_top_reviewers_skips_pr_author_and_deduplicates(self):
        def resolve_login(candidate):
            return {
                "alice@example.com": "alice-gh",
                "bob@example.com": "bob-gh",
                "carol@example.com": "carol-gh",
            }.get(candidate["author_email"])

        reviewers = MODULE.collect_top_reviewers(
            ["src/a.py", "src/b.py"],
            REVIEWER_COMMITS,
            pr_author_login="alice-gh",
            resolve_login=resolve_login,
        )

        self.assertEqual(
            [
                {
                    "login": "bob-gh",
                    "source_file": "src/a.py",
                    "author": "Bob",
                    "author_email": "bob@example.com",
                    "commit": "c3",
                    "commit_count": 1,
                },
            ],
            reviewers,
        )

    def test_collect_top_reviewers_skips_unresolved_candidates(self):
        def resolve_login(candidate):
            return {
                "carol@example.com": "carol-gh",
            }.get(candidate["author_email"])

        reviewers = MODULE.collect_top_reviewers(
            ["src/b.py"],
            REVIEWER_COMMITS,
            pr_author_login="pr-author",
            resolve_login=resolve_login,
        )

        self.assertEqual(
            [
                {
                    "login": "carol-gh",
                    "source_file": "src/b.py",
                    "author": "Carol",
                    "author_email": "carol@example.com",
                    "commit": "c4",
                    "commit_count": 1,
                }
            ],
            reviewers,
        )

    def test_render_reviewer_payload_outputs_json(self):
        payload = MODULE.render_reviewer_payload(
            [
                {
                    "login": "bob-gh",
                    "source_file": "src/a.py",
                    "author": "Bob",
                    "author_email": "bob@example.com",
                    "commit": "c3",
                    "commit_count": 1,
                }
            ]
        )

        self.assertEqual(
            {
                "reviewers": [
                    {
                        "login": "bob-gh",
                        "source_file": "src/a.py",
                        "author": "Bob",
                        "author_email": "bob@example.com",
                        "commit": "c3",
                        "commit_count": 1,
                    }
                ]
            },
            json.loads(payload),
        )


if __name__ == "__main__":
    unittest.main()
