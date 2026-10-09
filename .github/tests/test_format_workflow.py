import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _weekly_cron(workflow: str) -> tuple[int, int, int]:
    match = re.search(r"""- cron: ["'](\d+) (\d+) \* \* (\d+)["']""", workflow)

    assert match, "weekly cron not found"

    return tuple(int(group) for group in match.groups())


def test_format_sweep_runs_24h_before_the_weekly_prerelease() -> None:
    format_workflow = (REPO_ROOT / ".github/workflows/format.yml").read_text()
    release_workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text()

    minute, hour, weekday = _weekly_cron(format_workflow)
    release_minute, release_hour, release_weekday = _weekly_cron(release_workflow)

    assert (minute, hour) == (release_minute, release_hour)
    assert (weekday + 1) % 7 == release_weekday


def test_format_sweep_formats_the_whole_tree_and_pushes() -> None:
    workflow = (REPO_ROOT / ".github/workflows/format.yml").read_text()

    assert "clang-format-sweep:" in workflow
    assert "tools/format/clang_format.sh --all" in workflow
    assert "clang-format==18.1.3" in workflow
    assert "contents: write" in workflow
    assert "[skip changelog]" in workflow
