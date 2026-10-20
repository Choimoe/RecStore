import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "dockerfiles" / "git_ref.sh"


def run(cmd, cwd, **kwargs):
    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME", "RecStore Test")
    env.setdefault("GIT_AUTHOR_EMAIL", "recstore-test@example.com")
    env.setdefault("GIT_COMMITTER_NAME", "RecStore Test")
    env.setdefault("GIT_COMMITTER_EMAIL", "recstore-test@example.com")
    return subprocess.run(cmd, cwd=cwd, check=True, env=env, **kwargs)


class DockerGitRefTest(unittest.TestCase):
    def test_ensure_git_ref_fetches_missing_tag_in_shallow_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source"
            clone = tmp_path / "clone"

            source.mkdir()
            run(["git", "init", "-q", "-b", "main"], source)
            (source / "version.txt").write_text("tagged\n", encoding="utf-8")
            run(["git", "add", "version.txt"], source)
            run(["git", "commit", "-q", "-m", "tagged"], source)
            tagged_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
            run(["git", "tag", "v1.0.0"], source)
            (source / "version.txt").write_text("head\n", encoding="utf-8")
            run(["git", "commit", "-aq", "-m", "head"], source)

            run(["git", "clone", "-q", "--depth", "1", f"file://{source}", str(clone)], tmp_path)
            local_tag = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", "v1.0.0^{commit}"],
                cwd=clone,
                check=False,
            )
            self.assertNotEqual(local_tag.returncode, 0)

            script = f"source '{HELPER}'; ensure_git_ref v1.0.0"
            run(["bash", "-lc", script], clone)

            checked_out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip()
            self.assertEqual(checked_out, tagged_commit)


if __name__ == "__main__":
    unittest.main()
