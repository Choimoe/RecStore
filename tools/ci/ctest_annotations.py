#!/usr/bin/env python3
"""Emit GitHub Actions annotations for ctest results.

Usage: ctest_annotations.py <ctest-log-path>

Parses a ctest --output-on-failure log and emits workflow commands:

- one ``::error`` annotation per failed / timed-out / not-run test, with the
  test name, its lane, duration, and a few lines of failure context,
- a final ``::error`` summary if anything failed, or a ``::notice`` if
  everything passed, so the annotation area always states where ctest stood.

Annotation bodies are multi-line plain text (GitHub's Markdown rendering in
annotations is unreliable; titles do render backticks). When run inside
Actions, the summary also links the run and names the log artifact.

The script always exits 0; it never changes the outcome of a CI step.
"""

import os
import re
import sys

# GitHub renders at most 10 annotations per step.
MAX_ANNOTATIONS = 10
# Hard cap for one annotation body; keep well under the protocol limits.
MAX_MESSAGE_CHARS = 1200
# Lines of context around the failure line.
CONTEXT_BEFORE = 2
CONTEXT_AFTER = 4

# ctest pads the result column with dots:
#   " 5/54 Test  #5: name ....***Failed    1.2 sec"
RESULT_RE = re.compile(
    r"^\s*\d+/\d+\s+Test\s+#\d+:\s+(\S+?)[. ]*\*\*\*(\w[\w ]*?)\s+([\d.]+) sec"
)
START_RE = re.compile(r"^\s*Start\s+\d+:")
LANE_BANNER_RE = re.compile(r"^===== ctest lane: \S+ =====$")

LANES = ("unit", "python", "rdma_integration")

# Heuristics for the most useful line of a failure dump. Warnings that merely
# accompany a failure (e.g. Python ResourceWarning) are not root causes.
BORING_RE = re.compile(r"^ResourceWarning\b|^\s*::$")
INTERESTING_RE = re.compile(
    r"error|assert|fail|traceback|what\(\)|abort|exception|segmentation|"
    r"core dumped|missing|not found|cannot open",
    re.IGNORECASE,
)


def escape(message: str) -> str:
    """Escape a message for a GitHub workflow command data payload."""
    return (
        message.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )


def find_anchor(lines: list[str]) -> int:
    """Index of the most representative failure line, or -1.

    Prefers assertion-style lines (AssertionError, EXPECT_EQ, gtest
    "file:line: Failure") when present: those carry the actual values.
    Python tracebacks put them near the end, gtest near the top, so take the
    strongest match by score rather than first or last.
    """
    STRONG_RE = re.compile(
        r"AssertionError|AssertionFailedError|: Failure\b|EXPECT_|"
        r"Expected equality|what\(\):",
        re.IGNORECASE,
    )
    best, best_score = -1, 0
    for idx, line in enumerate(lines):
        stripped = line.rstrip()
        if not stripped or BORING_RE.search(stripped):
            continue
        score = 2 if STRONG_RE.search(stripped) else 0
        if score and INTERESTING_RE.search(stripped):
            score = 3
        if score > best_score:
            best, best_score = idx, score
    if best >= 0:
        return best
    for idx, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped and not BORING_RE.search(stripped):
            return idx
    return -1


def context_block(lines: list[str], anchor: int) -> str:
    """A few lines around the anchor, marked with '>' on the anchor line."""
    if anchor < 0:
        return "(no output captured)"
    lo = max(0, anchor - CONTEXT_BEFORE)
    hi = min(len(lines), anchor + CONTEXT_AFTER + 1)
    rendered = []
    for idx in range(lo, hi):
        stripped = lines[idx].rstrip()
        marker = ">" if idx == anchor else " "
        rendered.append(f"{marker} {stripped}")
    if lo > 0:
        rendered.insert(0, "  ...")
    if hi < len(lines):
        rendered.append("  ...")
    return "\n".join(rendered)


def collect_results(log_text: str) -> list[dict]:
    """One entry per finished test, in log order."""
    results = []
    current = None
    output_lines: list[str] = []

    def flush():
        nonlocal current, output_lines
        if current is not None:
            current["output"] = output_lines
            results.append(current)
        current = None
        output_lines = []

    for line in log_text.splitlines():
        match = RESULT_RE.match(line)
        if match:
            flush()
            current = {
                "name": match.group(1),
                "status": match.group(2).strip(),
                "duration": match.group(3),
                "output": [],
            }
            continue
        if START_RE.match(line) or LANE_BANNER_RE.match(line):
            flush()
            continue
        if current is not None:
            output_lines.append(line)
    flush()
    return results


def detect_lane(log_text: str) -> dict[str, str]:
    """Map test name -> lane.

    Two sources, in priority order:
    - The RECSTORE_CTEST_LANE environment variable: the workflow runs one
      annotation step per lane with this set, which is authoritative.
    - '===== ctest lane: X =====' banners, which tools/ci/reproduce.sh emits
      into its combined log for local runs.
    """
    env_lane = os.environ.get("RECSTORE_CTEST_LANE", "").strip()
    if env_lane:
        return {}
    lanes = {}
    current_lane = None
    for line in log_text.splitlines():
        banner = re.match(r"===== ctest lane: (\S+) =====", line)
        if banner:
            current_lane = banner.group(1)
            continue
        if current_lane:
            match = RESULT_RE.match(line)
            if match:
                lanes.setdefault(match.group(1), current_lane)
    return lanes


def run_link() -> str | None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return None
    server = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not (repo and run_id):
        return None
    return f"{server.replace('api.', '').replace('/repos', '')}/{repo}/actions/runs/{run_id}"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <ctest-log-path>", file=sys.stderr)
        return 0
    try:
        with open(sys.argv[1], "r", errors="replace") as handle:
            log_text = handle.read()
    except OSError as exc:
        # Annotation generation must never mask the real test result.
        print(f"::warning::could not read ctest log: {exc}")
        return 0

    results = collect_results(log_text)
    lanes = detect_lane(log_text)
    failures = [r for r in results if r["status"] != "Passed"]
    passed = [r for r in results if r["status"] == "Passed"]

    link = run_link()
    link_line = f"Run: {link}" if link else None

    if not failures:
        total = len(passed)
        print(f"::notice::ctest passed: {total} test(s) green")
        return 0

    print(f"::error::ctest: {len(failures)} test(s) failed or did not run "
          f"({len(passed)} passed)")
    for entry in failures[:MAX_ANNOTATIONS]:
        name = entry["name"]
        lane = lanes.get(name, "unknown lane")
        status = entry["status"]
        duration = entry["duration"]
        anchor = find_anchor(entry["output"])
        body = "\n".join(
            line
            for line in (
                f"Context ({status} after {duration}s):",
                context_block(entry["output"], anchor),
                "-----",
                *( [link_line] if link_line else [] ),
                'Full log: artifact "docker-image-test-logs" '
                "(runner/logs/ctest.log)",
            )
            if line
        )
        lane_label = f"{lane} lane" if lane != "unknown lane" else "unknown lane"
        title = f"{name} FAILED ({lane_label})"
        print(f"::error title={escape(title)}::{escape(body[:MAX_MESSAGE_CHARS])}")
    if len(failures) > MAX_ANNOTATIONS:
        remaining = len(failures) - MAX_ANNOTATIONS
        names = ", ".join(f["name"] for f in failures[MAX_ANNOTATIONS:])
        print(
            f"::error title=+{remaining} more failure(s)::"
            f"{escape(names[:MAX_MESSAGE_CHARS])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
