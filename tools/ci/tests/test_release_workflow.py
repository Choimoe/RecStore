from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pytorch_framework_build_is_not_gated_by_cuda() -> None:
    cmake = (REPO_ROOT / "src/framework/CMakeLists.txt").read_text()

    assert "add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)" in cmake
    assert "add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pytorch)" in cmake
    assert "_recstore_libtorch_requires_cuda" in cmake

    gpu_index = cmake.index("add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)")
    pytorch_index = cmake.index("add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pytorch)")
    first_endif_after_gpu = cmake.find("endif()", gpu_index)

    assert first_endif_after_gpu != -1
    assert pytorch_index > first_endif_after_gpu


def test_release_workflow_builds_cpu_libtorch_ops_package() -> None:
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text()

    assert "LIBTORCH_VARIANT=cpu" in workflow
    assert "cmake -DENABLE_CUDA=OFF" in workflow
    assert "bash tools/packaging/pack_artifact.sh" in workflow
    assert "build/lib/lib_recstore_ops.so" in workflow


def test_release_workflow_does_not_require_ycsb_artifact() -> None:
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text()

    assert "build/bin/ycsb" not in workflow
    assert "linux-x86_64-ycsb.tar.gz" not in workflow


def test_weekly_prerelease_can_be_manually_dispatched() -> None:
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text()

    assert "workflow_dispatch:" in workflow
    assert 'if [ "${GITHUB_EVENT_NAME}" = "push" ]; then' in workflow
    assert "Weekly Pre-release" in workflow


def test_weekly_release_notes_use_last_seven_days_from_head() -> None:
    workflow = (REPO_ROOT / ".github/workflows/release.yml").read_text()

    assert "release_notes_scope" in workflow
    assert "git rev-list -n 1 --before='7 days ago' HEAD" in workflow
    assert "RECENT_COMMITS=\"$(git rev-list --count \"${RELEASE_NOTES_SCOPE}\")\"" in workflow
    assert "needs.prepare-release.outputs.release_notes_scope" in workflow
