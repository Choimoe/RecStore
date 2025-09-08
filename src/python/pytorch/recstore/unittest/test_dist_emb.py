import torch
import unittest
import sys
import os

from ..DistEmb import DistEmbedding
from ..KVClient import get_kv_client
from ..optimizer import SparseSGD

class TestDistEmb(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class with a KV client and shared parameters."""
        cls.kv_client = get_kv_client()
        cls.embedding_dim = 16
        cls.learning_rate = 0.01

    def test_initialization_and_forward(self):
        """
        Tests that the DistEmbedding module initializes correctly in the backend
        and that the forward pass returns tensors of the correct shape and value.
        """
        num_embeddings = 100
        emb_name = "test_init_emb"

        dist_emb = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name)

        # Verify that the tensor metadata is registered in the client
        self.assertIn(emb_name, self.kv_client.data_name_list())
        dtype, shape = self.kv_client.get_data_meta(emb_name)
        self.assertEqual(shape, (num_embeddings, self.embedding_dim))
        self.assertEqual(dtype, torch.float32)

        # Test forward pass with some IDs
        input_ids = torch.tensor([10, 20, 30, 10], dtype=torch.int64)
        output_embs = dist_emb(input_ids)

        # Check output shape and default zero-initialized values
        self.assertEqual(output_embs.shape, (len(input_ids), self.embedding_dim))
        self.assertTrue(torch.all(output_embs == 0))

    def test_backward_and_optimizer_step(self):
        """
        Tests the new autograd workflow: backward() traces gradients, and
        optimizer.step() applies the update.
        """
        num_embeddings = 50
        emb_name = "test_update_emb"

        # Custom initializer to create non-zero embeddings
        def initializer(shape, dtype):
            return torch.ones(shape, dtype=dtype) * 0.5

        dist_emb = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name, init_func=initializer)
        optimizer = SparseSGD([dist_emb], lr=self.learning_rate)
        
        input_ids = torch.tensor([5, 15, 25], dtype=torch.int64)
        
        # 1. Check initial values
        initial_values = self.kv_client.pull(emb_name, input_ids)
        self.assertTrue(torch.all(initial_values == 0.5))

        # 2. Perform forward and backward passes
        optimizer.zero_grad() # Resets the trace
        output_embs = dist_emb(input_ids)
        self.assertTrue(output_embs.requires_grad)
        
        loss = output_embs.sum()
        loss.backward() # This should only populate the trace, not update weights

        # 3. Verify that weights have NOT changed after backward()
        values_after_backward = self.kv_client.pull(emb_name, input_ids)
        self.assertTrue(
            torch.allclose(initial_values, values_after_backward),
            "Values should not change after backward(), only after optimizer.step()"
        )
        self.assertGreater(len(dist_emb._trace), 0, "Trace should be populated after backward()")

        # 4. Perform optimizer step and verify weights HAVE changed
        optimizer.step()
        updated_values = self.kv_client.pull(emb_name, input_ids)
        
        expected_gradient = torch.ones_like(initial_values)
        expected_values = initial_values - (self.learning_rate * expected_gradient)

        self.assertTrue(
            torch.allclose(updated_values, expected_values),
            f"Update failed! Expected:\n{expected_values}\nGot:\n{updated_values}"
        )
        
    def test_persistence_with_same_name(self):
        """
        Tests that embedding data persists and can be accessed by a new
        DistEmbedding instance with the same name. This test is self-contained.
        """
        num_embeddings = 60
        emb_name = "test_persistence_emb" 

        # --- First instance: initialize and update ---
        dist_emb_1 = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name)
        optimizer = SparseSGD([dist_emb_1], lr=self.learning_rate)
        input_ids = torch.tensor([10, 20, 30], dtype=torch.int64)

        optimizer.zero_grad()
        loss = dist_emb_1(input_ids).sum()
        loss.backward()
        optimizer.step()

        # Calculate the expected values after the update (initial was zeros)
        expected_values = 0 - (self.learning_rate * torch.ones((len(input_ids), self.embedding_dim)))
        
        # --- Second instance: should load the updated data ---
        dist_emb_2 = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name)
        values_from_new_instance = dist_emb_2(input_ids)
        
        self.assertTrue(
            torch.allclose(values_from_new_instance.detach(), expected_values),
            "New instance failed to access persisted, updated values from the first instance."
        )

    def test_update_with_duplicate_ids(self):
        """
        Tests that the SparseSGD optimizer correctly accumulates gradients
        for duplicate IDs within a batch.
        """
        num_embeddings = 70
        emb_name = "test_duplicate_ids_emb"

        dist_emb = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name)
        optimizer = SparseSGD([dist_emb], lr=self.learning_rate)

        # ID 1 appears 3 times, ID 2 appears 2 times, ID 3 appears once
        input_ids = torch.tensor([1, 2, 1, 3, 2, 1], dtype=torch.int64)
        
        optimizer.zero_grad()
        output_embs = dist_emb(input_ids)
        loss = output_embs.sum()
        loss.backward()
        optimizer.step() # Optimizer should handle gradient aggregation

        # Check the updated values
        updated_values_for_id_1 = self.kv_client.pull(emb_name, torch.tensor([1]))
        updated_values_for_id_2 = self.kv_client.pull(emb_name, torch.tensor([2]))

        # The gradient for each lookup is 1. The optimizer should sum them up.
        expected_grad_for_id_1 = torch.ones((1, self.embedding_dim)) * 3 # Appeared 3 times
        expected_grad_for_id_2 = torch.ones((1, self.embedding_dim)) * 2 # Appeared 2 times

        # Initial values were 0
        expected_updated_val_1 = 0 - (self.learning_rate * expected_grad_for_id_1)
        expected_updated_val_2 = 0 - (self.learning_rate * expected_grad_for_id_2)

        self.assertTrue(torch.allclose(updated_values_for_id_1, expected_updated_val_1))
        self.assertTrue(torch.allclose(updated_values_for_id_2, expected_updated_val_2))


if __name__ == '__main__':
    # Add project root to path to allow relative imports
    # This is a common pattern for running tests directly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    unittest.main()
