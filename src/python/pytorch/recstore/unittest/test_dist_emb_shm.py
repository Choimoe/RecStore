import torch
import torch.optim as optim
import unittest
import os
import sys

from ..DistEmb import DistEmbedding
from ..KVClient import get_kv_client

# ===================================================================
# Multi-process test runner
# ===================================================================

def run_tests_for_rank(rank, world_size):
    """
    Load and run all test cases for a single rank.
    unittest.main() does not work with mpirun, so we simulate its behavior.
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    
    print(f"\n=============== [Rank {rank}/{world_size}] Running All Tests ===============\n")
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)


# ===================================================================
# Multi-process test cases
# ===================================================================

class TestDistEmbSHM(unittest.TestCase):
    """
    A test suite containing multiple multi-process test cases.
    Each 'test_*' method is an independent, complete test scenario.
    """
    
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))

    @classmethod
    def setUpClass(cls):
        """Executed once per process before all tests."""
        os.environ["RECSTORE_BACKEND_TYPE"] = "SHM"
        cls.kv_client = get_kv_client()
        cls.learning_rate = 1.0

    def setUp(self):
        """Executed before each test method."""
        print(f"\n--- [Rank {self.rank}] Starting test: {self._testMethodName} ---")
        # Reset embedding dimension to allow different dimensions in different tests
        self.kv_client.reset_embedding_dimension()
        # Use barrier to ensure all processes start each test in sync
        self.kv_client.barrier()

    def tearDown(self):
        """Executed after each test method."""
        self.kv_client.barrier()
        print(f"--- [Rank {self.rank}] Finished test: {self._testMethodName} ---")

    def test_initialization_and_sync(self):
        """
        Test 1: Initialization and synchronization
        - Rank 0 initializes an embedding.
        - All other ranks verify they can read the correct initial value.
        """
        num_embeddings = 20
        embedding_dim = 8
        # Use rank and method name to ensure unique emb_name for each test
        emb_name = f"test_init_sync_emb_{self._testMethodName}"

        if self.rank == 0:
            def initializer(shape, dtype):
                return torch.full(shape, dtype=dtype, fill_value=float(self.world_size))
            DistEmbedding(num_embeddings, embedding_dim, name=emb_name, init_func=initializer, persistent=True)

        self.kv_client.barrier()

        dist_emb = DistEmbedding(num_embeddings, embedding_dim, name=emb_name, persistent=True)
        ids_to_read = torch.tensor([0, 5, 15], dtype=torch.int64)
        read_values = dist_emb.weight[ids_to_read]

        expected_values = torch.full((len(ids_to_read), embedding_dim), fill_value=float(self.world_size))
        self.assertTrue(torch.allclose(read_values, expected_values))

    def test_distinct_updates_and_verification(self):
        """
        Test 2: Distinct updates and global verification (core test)
        - Each rank updates a completely different ID subset.
        - After update, each rank verifies it can read all other ranks' updates.
        """
        num_embeddings = self.world_size * 10
        embedding_dim = 4
        emb_name = f"test_distinct_updates_emb_{self._testMethodName}"

        model = DistEmbedding(num_embeddings, embedding_dim, name=emb_name, persistent=True)
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        ids_to_update = torch.tensor([self.rank * 10 + i for i in range(2)], dtype=torch.int64)
        
        optimizer.zero_grad()
        embs = model(ids_to_update)
        loss = ((embs - 1.0) ** 2).sum()
        loss.backward()
        optimizer.step()

        expected_value = 2.0
        for i in range(self.world_size):
            ids_to_check = torch.tensor([i * 10 + j for j in range(2)], dtype=torch.int64)
            checked_embs = model.weight[ids_to_check]
            expected_embs = torch.full_like(checked_embs, expected_value)
            self.assertTrue(
                torch.allclose(checked_embs, expected_embs),
                f"[Rank {self.rank}] Failed to verify update from Rank {i}. Got {checked_embs}"
            )

    def test_overlapping_updates(self):
        """
        Test 3: Overlapping updates
        - All ranks update the same ID (ID 5).
        - Verify the final value of ID 5 is the sum of all gradient updates.
        """
        if self.world_size < 2:
            self.skipTest("This test requires at least 2 processes.")

        num_embeddings = 10
        embedding_dim = 2
        emb_name = f"test_overlapping_emb_{self._testMethodName}"

        model = DistEmbedding(num_embeddings, embedding_dim, name=emb_name, persistent=True)
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        ids_to_update = torch.tensor([5], dtype=torch.int64)

        optimizer.zero_grad()
        embs = model(ids_to_update)
        loss = ((embs - 1.0) ** 2).sum()
        loss.backward()
        optimizer.step()

        expected_value = 2.0 * self.world_size
        final_emb = model.weight[torch.tensor([5])]
        expected_emb = torch.full_like(final_emb, expected_value)
        self.assertTrue(
            torch.allclose(final_emb, expected_emb),
            f"Expected: {expected_emb}, Got: {final_emb}"
        )

    def test_empty_update_from_one_rank(self):
        """
        Test 4: Partial empty update
        - Only rank 0 has a gradient update, other ranks send empty updates.
        - Verify the system handles this correctly and only rank 0's update is applied.
        """
        if self.world_size < 2:
            self.skipTest("This test requires at least 2 processes.")

        num_embeddings = 10
        embedding_dim = 2
        emb_name = f"test_empty_update_emb_{self._testMethodName}"

        model = DistEmbedding(num_embeddings, embedding_dim, name=emb_name, persistent=True)
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()

        if self.rank == 0:
            ids_to_update = torch.tensor([1], dtype=torch.int64)
            embs = model(ids_to_update)
            loss = ((embs - 1.0) ** 2).sum()
            loss.backward()
        
        optimizer.step()

        updated_emb_1 = model.weight[torch.tensor([1])]
        self.assertTrue(torch.allclose(updated_emb_1, torch.full_like(updated_emb_1, 2.0)))

        updated_emb_3 = model.weight[torch.tensor([3])]
        self.assertTrue(torch.allclose(updated_emb_3, torch.full_like(updated_emb_3, 0.0)))


if __name__ == "__main__":
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
    run_tests_for_rank(rank, world_size)
