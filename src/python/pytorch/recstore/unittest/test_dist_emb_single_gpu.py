import torch
import unittest
import os
import sys

# Set the backend type to GPU before importing any recstore modules
os.environ['RECSTORE_BACKEND_TYPE'] = 'GPU'

# Add project root to path to allow direct execution
# This allows running the test script directly, e.g., for debugging in an IDE
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# It's good practice to import after setting the environment variable
from ..DistEmb import DistEmbedding
from ..KVClient import get_kv_client

# Conditionally skip all tests in this file if CUDA is not available
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available, skipping GPU tests")
class TestDistEmbSingleGPU(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the test class. This runs once before all tests.
        Initializes the KV client and sets common parameters.
        """
        cls.device = torch.device("cuda:0")
        cls.kv_client = get_kv_client()
        cls.embedding_dim = 16
        cls.learning_rate = 0.01
        print("\n--- Running Comprehensive GPU Backend Tests ---")

    def test_a_initialization_and_forward(self):
        """
        Tests that the embedding layer can be initialized and a forward pass
        on the GPU returns a tensor on the correct device with zero values.
        """
        num_embeddings = 100
        emb_name = "test_gpu_init_emb"

        dist_emb = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name)
        input_ids_gpu = torch.tensor([10, 20, 99, 10], dtype=torch.int64).to(self.device)

        output_embs = dist_emb(input_ids_gpu)

        self.assertEqual(output_embs.device.type, 'cuda', "Output tensor is not on CUDA device")
        self.assertEqual(output_embs.shape, (len(input_ids_gpu), self.embedding_dim))
        self.assertTrue(torch.all(output_embs == 0), "Initial embeddings should be all zeros")

    def test_b_backward_and_update(self):
        """
        Tests a full forward, backward, and update cycle on the GPU with a custom initializer.
        """
        num_embeddings = 50
        emb_name = "test_gpu_update_emb"

        def initializer(shape, dtype):
            return torch.ones(shape, dtype=dtype) * 0.5

        dist_emb = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name, init_func=initializer)
        input_ids_gpu = torch.tensor([5, 15, 25], dtype=torch.int64).to(self.device)
        
        # 1. Forward pass
        output_embs = dist_emb(input_ids_gpu)
        self.assertTrue(output_embs.requires_grad)
        
        # 2. Backward pass
        loss = output_embs.sum()
        loss.backward()

        # 3. Check updated values in the store
        updated_values_gpu = self.kv_client.pull(emb_name, input_ids_gpu)

        initial_values = torch.ones_like(updated_values_gpu) * 0.5
        expected_gradient = torch.ones_like(initial_values)
        expected_values = initial_values - (self.learning_rate * expected_gradient)

        self.assertTrue(
            torch.allclose(updated_values_gpu, expected_values),
            f"Update failed on GPU! Expected:\n{expected_values}\nGot:\n{updated_values_gpu}"
        )

    def test_c_persistence_with_same_name(self):
        """
        Tests that creating a new embedding instance with the same name correctly
        accesses the previously updated, persistent state.
        """
        num_embeddings = 50
        emb_name = "test_gpu_update_emb" # Same name as in test_b

        # Create a new instance, it should not re-initialize the data
        new_dist_emb_instance = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name)
        input_ids_gpu = torch.tensor([5, 15, 25], dtype=torch.int64).to(self.device)
        
        # Perform a forward pass to get the values
        values_from_new_instance = new_dist_emb_instance(input_ids_gpu)
        
        # The expected values are the final values from the previous test
        initial_values = torch.ones((3, self.embedding_dim), dtype=torch.float32, device=self.device) * 0.5
        expected_gradient = torch.ones_like(initial_values)
        expected_persisted_values = initial_values - (self.learning_rate * expected_gradient)

        self.assertTrue(
            torch.allclose(values_from_new_instance.detach(), expected_persisted_values),
            "New instance failed to access persisted, updated values."
        )

    def test_d_gradient_accumulation_with_duplicate_ids(self):
        """
        Crucial test: Verifies that gradients for duplicate IDs in a batch are
        correctly accumulated before the update operation.
        """
        num_embeddings = 20
        emb_name = "test_gpu_duplicate_ids_emb"

        dist_emb = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name)
        # ID 1 appears 3 times, ID 8 appears 2 times.
        input_ids_gpu = torch.tensor([1, 8, 3, 1, 8, 1], dtype=torch.int64).to(self.device)
        
        output_embs = dist_emb(input_ids_gpu)
        loss = output_embs.sum()
        loss.backward()

        # Check the updated values for the unique IDs that appeared
        unique_ids_gpu = torch.tensor([1, 3, 8], dtype=torch.int64).to(self.device)
        updated_values = self.kv_client.pull(emb_name, unique_ids_gpu)

        # Expected values are calculated based on accumulated gradients
        # Initial value is 0. Update rule is: new = old - lr * grad
        # grad for id 1 is 3 (since it appears 3 times)
        # grad for id 3 is 1
        # grad for id 8 is 2
        expected_val_1 = 0 - self.learning_rate * 3
        expected_val_3 = 0 - self.learning_rate * 1
        expected_val_8 = 0 - self.learning_rate * 2
        
        self.assertTrue(torch.allclose(updated_values[0], torch.full_like(updated_values[0], expected_val_1)))
        self.assertTrue(torch.allclose(updated_values[1], torch.full_like(updated_values[1], expected_val_3)))
        self.assertTrue(torch.allclose(updated_values[2], torch.full_like(updated_values[2], expected_val_8)))

    def test_e_empty_input(self):
        """
        Tests the behavior of the layer when provided with an empty input tensor.
        It should not crash and should return a correctly shaped empty tensor.
        """
        num_embeddings = 10
        emb_name = "test_gpu_empty_input_emb"
        dist_emb = DistEmbedding(num_embeddings, self.embedding_dim, name=emb_name)

        input_ids_gpu = torch.tensor([], dtype=torch.int64).to(self.device)
        output_embs = dist_emb(input_ids_gpu)

        self.assertEqual(output_embs.shape, (0, self.embedding_dim))
        # Ensure backward pass on empty output doesn't crash
        loss = output_embs.sum()
        loss.backward()


if __name__ == '__main__':
    unittest.main()
