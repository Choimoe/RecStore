import pathlib
import unittest


class InitDlrmScriptTest(unittest.TestCase):
    def test_builds_torchrec_stack_from_source_against_installed_torch(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        script = (repo_root / 'dockerfiles' / 'init_dlrm.sh').read_text()

        self.assertNotIn('pip install fbgemm-gpu==1.0', script)
        self.assertNotIn('pip install torchrec==1.0', script)
        self.assertIn('https://github.com/pytorch/FBGEMM.git', script)
        self.assertIn('https://github.com/pytorch/torchrec.git', script)
        self.assertIn('asynchronous_complete_cumsum', script)
        self.assertIn('setuptools-git-versioning', script)
        self.assertIn('Using ninja binary', script)
        self.assertIn('CMAKE_MAKE_PROGRAM', script)
        self.assertIn('python_shims', script)
        self.assertIn('setup.py bdist_wheel', script)
        self.assertIn('--no-deps dist/*.whl', script)
        self.assertIn('patchelf --add-needed libtbb.so.12', script)
        self.assertIn('importlib.util.find_spec("fbgemm_gpu")', script)
        self.assertIn('export USE_CXX11_ABI="${TORCH_CXX11_ABI}"', script)
        self.assertIn('-D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}', script)
        self.assertNotIn('/home/shq/', script)
        self.assertIn('FBGEMM_REF="${FBGEMM_REF:-v1.1.2}"', script)
        self.assertIn('TORCHREC_REF="${TORCHREC_REF:-v1.1.0}"', script)

    def test_skips_source_build_when_cuda_toolkit_is_unavailable(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        script = (repo_root / 'dockerfiles' / 'init_dlrm.sh').read_text()

        self.assertIn('command -v nvcc >/dev/null 2>&1', script)
        self.assertIn('Skipping TorchRec/FBGEMM source build because CUDA toolkit is not available', script)
        self.assertIn('Detected CPU-only torch install; skipping TorchRec/FBGEMM source build', script)


if __name__ == '__main__':
    unittest.main()
