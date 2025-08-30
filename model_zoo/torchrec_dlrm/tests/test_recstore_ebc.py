import argparse
import os
import sys
import torch
from torchrec import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig

RECSTORE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src'))
if RECSTORE_PATH not in sys.path:
    sys.path.insert(0, RECSTORE_PATH)

from python.pytorch.torchrec.EmbeddingBag import RecStoreEmbeddingBagCollection
from python.pytorch.recstore.KVClient import get_kv_client

from python.pytorch.recstore.optimizer import SparseSGD

# --- Constants ---
LEARNING_RATE = 0.01
NUM_TABLES = 1

def get_eb_configs(
    num_embeddings: int,
    embedding_dim: int,
) -> list:
    """
    Generates a list of EmbeddingBagConfig objects for a single table.
    """
    return [
        EmbeddingBagConfig(
            name="table_0",
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=["feature_0"],
        )
    ]

def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, label: str, atol=1e-6) -> bool:
    """
    Compares two tensors for near-equality and prints a detailed result.
    """
    print(f"\n----- Comparing '{label}' -----")
    t1, t2 = tensor1.detach(), tensor2.detach()
    
    print(f"  - Shape of Tensor 1 (Expected): {t1.shape}")
    print(f"  - Shape of Tensor 2 (Actual):   {t2.shape}")

    if t1.shape != t2.shape:
        print(f"❌ FAILURE: {label} outputs have MISMATCHED SHAPES.")
        return False

    are_close = torch.allclose(t1, t2, atol=atol)
    if are_close:
        print(f"✅ SUCCESS: {label} outputs are numerically aligned.")
    else:
        print(f"❌ FAILURE: {label} outputs are NOT aligned.")
        max_diff = (t1 - t2).abs().max().item()
        print(f"   - Max absolute difference: {max_diff:.8f}")
        print(f"   - Sliced Tensor 1 (Expected): \n{t1.flatten()[:8]}")
        print(f"   - Sliced Tensor 2 (Actual):   \n{t2.flatten()[:8]}")
    return are_close

def main(args):
    # --- 1. Setup Environment ---
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Running SINGLE TABLE precision test on device: {device}")

    # --- 2. Create Configurations ---
    eb_configs = get_eb_configs(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
    )
    
    recstore_eb_configs_dict = [
        {
            "name": c.name,
            "embedding_dim": c.embedding_dim,
            "num_embeddings": c.num_embeddings,
            "feature_names": c.feature_names,
        }
        for c in eb_configs
    ]

    # --- 3. Instantiate Models ---
    print("\nInstantiating standard torchrec.EmbeddingBagCollection (Ground Truth)...")
    standard_ebc = EmbeddingBagCollection(tables=eb_configs, device=device)
    
    print("Instantiating custom RecStoreEmbeddingBagCollection...")
    recstore_ebc = RecStoreEmbeddingBagCollection(embedding_bag_configs=recstore_eb_configs_dict)
    recstore_ebc.to(device)

    # --- 4. Synchronize Initial Weights ---
    print("\n--- Initializing and Synchronizing Weights ---")
    kv_client = get_kv_client()
    with torch.no_grad():
        config = eb_configs[0]
        table_weights = standard_ebc.state_dict()[f"embedding_bags.{config.name}.weight"]
        all_keys = torch.arange(config.num_embeddings, dtype=torch.int64)
        print(f"Synchronizing weights for the single table '{config.name}'...")
        kv_client.push(name=config.name, ids=all_keys, data=table_weights.cpu())

    # --- 5. Setup Optimizers ---
    print(f"\nSetting up optimizers with LR = {LEARNING_RATE}.")
    standard_optimizer = torch.optim.SGD(standard_ebc.parameters(), lr=LEARNING_RATE)
    sparse_optimizer = SparseSGD([recstore_ebc], lr=LEARNING_RATE)

    # --- 6. Create a Deterministic Input Batch ---
    batch = KeyedJaggedTensor.from_lengths_sync(
        keys=["feature_0"],
        values=torch.tensor([10, 20, 30, 5], device=device, dtype=torch.int64),
        lengths=torch.tensor([3, 1], device=device, dtype=torch.int32), # B=2
    )
    print(f"\nCreated a deterministic batch for a single feature and batch size of 2.")

    # --- 7. Perform End-to-End Precision Test ---
    all_tests_ok = True

    # a) Forward Pass Comparison
    print("\n" + "="*30)
    print("STEP 1: FORWARD PASS")
    print("="*30)
    standard_output_kt = standard_ebc(batch)
    recstore_output_kt = recstore_ebc(batch)
    
    forward_pass_ok = compare_tensors(
        standard_output_kt.values(), 
        recstore_output_kt.values(), 
        "Forward Pass Output"
    )
    if not forward_pass_ok:
        print("🔥 Halting test: Forward pass failed. Please fix before checking gradients.")
        sys.exit(1)

    # b) Gradient Calculation (Implicitly tested by weight update)
    dummy_loss_standard = standard_output_kt.values().sum()
    dummy_loss_recstore = recstore_output_kt.values().sum()
    
    standard_optimizer.zero_grad()
    sparse_optimizer.zero_grad()

    dummy_loss_standard.backward()
    dummy_loss_recstore.backward()

    # c) Weight Update
    print("\n" + "="*30)
    print("STEP 2: WEIGHT UPDATE")
    print("="*30)
    standard_optimizer.step()
    sparse_optimizer.step()
    print("Standard EBC weights updated via standard optimizer.")
    print("RecStore EBC weights updated via sparse optimizer.")

    # d) Final Weight Comparison
    print("\n" + "="*30)
    print("STEP 3: FINAL WEIGHT COMPARISON")
    print("="*30)
    with torch.no_grad():
        config = eb_configs[0]
        updated_standard_weights = standard_ebc.state_dict()[f"embedding_bags.{config.name}.weight"]
        all_keys = torch.arange(config.num_embeddings, dtype=torch.int64)
        updated_recstore_weights = kv_client.pull(name=config.name, ids=all_keys).to(device)
        
        weights_ok = compare_tensors(
            updated_standard_weights, 
            updated_recstore_weights, 
            f"Updated Weights for '{config.name}'"
        )
        if not weights_ok:
            all_tests_ok = False

    # --- 8. Final Summary ---
    print("\n" + "#"*30)
    print("### SINGLE-TABLE TEST SUMMARY ###")
    print("#"*30)
    if all_tests_ok:
        print("🎉🎉🎉 All precision tests passed! The core logic for a single table is correct.")
        print("Next step: Implement multi-table support in your C++ backend.")
    else:
        print("🔥🔥🔥 Precision test failed. The issue exists even with a single table.")
        print("   - If 'Forward Pass' failed: Check `EmbeddingBag.py` forward logic (pooling).")
        print("   - If 'Updated Weights' failed: Check `EmbeddingBag.py` backward logic and `op.cc` EmbUpdate implementation.")
        print(f"   - CRITICAL: Ensure the learning rate in your optimizer ({MOCK_BACKEND_LR}) matches the one in `op.cc`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-table precision test for RecStore EBC.")
    parser.add_argument("--num-embeddings", type=int, default=100, help="Number of embeddings per table.")
    parser.add_argument("--embedding-dim", type=int, default=16, help="Dimension of embeddings.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--cpu", action="store_true", help="Force test to run on CPU.")
    
    args = parser.parse_args()
    main(args)
