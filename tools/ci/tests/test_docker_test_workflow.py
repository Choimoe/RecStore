from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_docker_test_uses_matching_main_image_without_fallback() -> None:
    workflow = (REPO_ROOT / ".github/workflows/test.yml").read_text()

    assert "branches: [main]" in workflow
    assert "github.event.workflow_run.head_sha" in workflow
    assert "needs: detect-image-change" in workflow
    assert "defer Docker Test until Docker Build and Publish completes" in workflow
    assert "try_pull" not in workflow
    assert "PRIMARY_SHA_TAG" not in workflow


def test_docker_image_declares_the_schema_test_dependency() -> None:
    dockerfile = (REPO_ROOT / "dockerfiles/Dockerfile.recstore").read_text()

    assert "jsonschema numpy tqdm" in dockerfile


def test_docker_failure_notifications_are_serialized_for_one_tracker() -> None:
    workflow = (
        REPO_ROOT / ".github/workflows/ci-failure-notifications.yml"
    ).read_text()

    assert "'Docker Test' && 'tracker'" in workflow
    assert "cancel-in-progress: false" in workflow
