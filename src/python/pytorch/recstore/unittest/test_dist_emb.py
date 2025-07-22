import torch
import unittest
import sys
import os

from ..DistEmb import DistEmbedding
from ..KVClient import get_kv_client

class TestDistEmb(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.kv_client = get_kv_client()
        cls.embedding_dim = 16
        cls.learning_rate = 0.01

    def test_a_initialization_and_forward(self):
        num_embeddings = 100
        emb_name = "test_init_emb"

        dist_emb = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name)

        self.assertIn(emb_name, self.kv_client.data_name_list())
        dtype, shape = self.kv_client.get_data_meta(emb_name)
        self.assertEqual(shape, (num_embeddings, self.embedding_dim))
        self.assertEqual(dtype, torch.float32)

        input_ids = torch.tensor([10, 20, 30, 10], dtype=torch.int64)
        output_embs = dist_emb(input_ids)

        self.assertEqual(output_embs.shape, (len(input_ids), self.embedding_dim))
        self.assertTrue(torch.all(output_embs == 0))

    def test_b_backward_and_update(self):
        num_embeddings = 50
        emb_name = "test_update_emb"

        def initializer(shape, dtype):
            return torch.ones(shape, dtype=dtype) * 0.5

        dist_emb = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name, init_func=initializer)
        
        input_ids = torch.tensor([5, 15, 25], dtype=torch.int64)
        
        initial_values = self.kv_client.pull(emb_name, input_ids)
        self.assertTrue(torch.all(initial_values == 0.5))

        output_embs = dist_emb(input_ids)
        self.assertTrue(output_embs.requires_grad)
        
        loss = output_embs.sum()
        loss.backward()

        updated_values = self.kv_client.pull(emb_name, input_ids)
        
        expected_gradient = torch.ones_like(initial_values)
        expected_values = initial_values - (self.learning_rate * expected_gradient)

        self.assertTrue(
            torch.allclose(updated_values, expected_values),
            f"Update failed! Expected:\n{expected_values}\nGot:\n{updated_values}"
        )

    def test_c_persistence_with_same_name(self):
        num_embeddings = 50
        emb_name = "test_update_emb" 

        new_dist_emb_instance = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name)

        input_ids = torch.tensor([5, 15, 25], dtype=torch.int64)
        
        values_from_new_instance = new_dist_emb_instance(input_ids)
        
        initial_values = torch.ones((3, self.embedding_dim), dtype=torch.float32) * 0.5
        expected_gradient = torch.ones_like(initial_values)
        expected_values = initial_values - (self.learning_rate * expected_gradient)

        self.assertTrue(
            torch.allclose(values_from_new_instance.detach(), expected_values),
            "New instance failed to access persisted, updated values."
        )

    def test_d_lookup_with_duplicate_ids(self):
        num_embeddings = 70
        emb_name = "test_duplicate_ids_emb"

        dist_emb = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name)

        input_ids = torch.tensor([1, 2, 1, 3, 2, 1], dtype=torch.int64)
        
        output_embs = dist_emb(input_ids)
        self.assertEqual(output_embs.shape, (len(input_ids), self.embedding_dim))

        loss = output_embs.sum()
        loss.backward()

        updated_values_for_id_1 = self.kv_client.pull(emb_name, torch.tensor([1]))
        updated_values_for_id_2 = self.kv_client.pull(emb_name, torch.tensor([2]))

        expected_grad_for_id_1 = torch.ones((1, self.embedding_dim)) * 3
        expected_grad_for_id_2 = torch.ones((1, self.embedding_dim)) * 2

        expected_updated_val_1 = 0 - (self.learning_rate * expected_grad_for_id_1)
        expected_updated_val_2 = 0 - (self.learning_rate * expected_grad_for_id_2)

        self.assertTrue(torch.allclose(updated_values_for_id_1, expected_updated_val_1))
        self.assertTrue(torch.allclose(updated_values_for_id_2, expected_updated_val_2))

if __name__ == '__main__':
    unittest.main()
