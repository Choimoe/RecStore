import unittest

import torch
from torchrec.sparse.jagged_tensor import KeyedTensor

from model_zoo.rs_demo.data.dlrm_source import build_kjt_batch_from_dense_sparse_labels
from model_zoo.rs_demo.models.utils import reshape_torchrec_embeddings_for_dlrm


class TestDlrmInputLayout(unittest.TestCase):
    def test_fixed_bags_preserve_feature_order_and_partial_batches(self):
        for rows in (0, 1, 7):
            sparse = torch.arange(rows * 26).view(rows, 26)
            dense = torch.ones(rows, 13)
            returned, features = build_kjt_batch_from_dense_sparse_labels(
                dense, sparse, torch.zeros(rows)
            )
            self.assertIs(returned, dense)
            self.assertEqual(features.stride(), rows)
            self.assertEqual(features.length_per_key(), [rows] * 26)
            for index, key in enumerate(features.keys()):
                torch.testing.assert_close(features[key].values(), sparse[:, index])
                torch.testing.assert_close(features[key].lengths(), torch.ones(rows, dtype=torch.int32))

    def test_view_and_reordered_fallback_preserve_outputs_and_gradients(self):
        for keys in (["a", "b"], ["b", "a"], ["b"]):
            values = torch.randn(3, 8, requires_grad=True)
            keyed = KeyedTensor(keys=["a", "b"], length_per_key=[4, 4], values=values)
            actual = reshape_torchrec_embeddings_for_dlrm(keyed, keys, torch)
            expected = torch.stack([keyed[key] for key in keys], dim=1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            grad = torch.randn_like(actual)
            left = torch.autograd.grad(actual, values, grad, retain_graph=True)[0]
            right = torch.autograd.grad(expected, values, grad)[0]
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            if keys == ["a", "b"]:
                self.assertEqual(actual.data_ptr(), values.data_ptr())
        mapping = {"a": torch.ones(2, 4), "b": torch.zeros(2, 4)}
        torch.testing.assert_close(
            reshape_torchrec_embeddings_for_dlrm(mapping, ["b", "a"], torch),
            torch.stack([mapping["b"], mapping["a"]], dim=1),
        )


if __name__ == "__main__":
    unittest.main()
