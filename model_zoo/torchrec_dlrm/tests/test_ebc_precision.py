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
NUM_TEST_ROUNDS = 10


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

    if t1.shape != t2.shape:
        print(f"❌ FAILURE: {label} outputs have MISMATCHED SHAPES.")
        print(f"  - Shape of Tensor 1 (Expected): {t1.shape}")
        print(f"  - Shape of Tensor 2 (Actual):   {t2.shape}")
        return False

    are_close = torch.allclose(t1, t2, atol=atol)
    if are_close:
        print(f"   - Sliced Tensor 1 (Expected): \n{t1.flatten()[:8]}")
        print(f"   - Sliced Tensor 2 (Actual):   \n{t2.flatten()[:8]}")
        print(f"✅ SUCCESS: {label} outputs are numerically aligned.")
    else:
        print(f"❌ FAILURE: {label} outputs are NOT aligned.")
        max_diff = (t1 - t2).abs().max().item()
        print(f"   - Max absolute difference: {max_diff:.8f}")
        if max_diff > 1e-5:
            print(f"   - Sliced Tensor 1 (Expected): \n{t1.flatten()[:8]}")
            print(f"   - Sliced Tensor 2 (Actual):   \n{t2.flatten()[:8]}")
    return are_close

def generate_random_batch(num_embeddings, batch_size, device):
    """
    Generates a random KeyedJaggedTensor batch.
    """
    avg_len = max(1, (num_embeddings // batch_size) // 2)
    lengths = torch.randint(1, avg_len * 2, (batch_size,), device=device, dtype=torch.int32)
    values = torch.randint(0, num_embeddings, (lengths.sum().item(),), device=device, dtype=torch.int64)
    return KeyedJaggedTensor.from_lengths_sync(
        keys=["feature_0"],
        values=values,
        lengths=lengths,
    )


def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Running SINGLE TABLE precision test on device: {device} for {NUM_TEST_ROUNDS} rounds.")

    eb_configs = get_eb_configs(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
    )
    
    recstore_eb_configs_dict = [
        {"name": c.name, "embedding_dim": c.embedding_dim, "num_embeddings": c.num_embeddings, "feature_names": c.feature_names}
        for c in eb_configs
    ]

    print("\nInstantiating standard torchrec.EmbeddingBagCollection (Ground Truth)...")
    standard_ebc = EmbeddingBagCollection(tables=eb_configs, device=device)
    
    print("Instantiating custom RecStoreEmbeddingBagCollection...")
    recstore_ebc = RecStoreEmbeddingBagCollection(embedding_bag_configs=recstore_eb_configs_dict).to(device)

    print("\n--- Initializing and Synchronizing Weights ---")
    kv_client = get_kv_client()
    with torch.no_grad():
        config = eb_configs[0]
        initial_weights = standard_ebc.state_dict()[f"embedding_bags.{config.name}.weight"]
        all_keys = torch.arange(config.num_embeddings, dtype=torch.int64)
        kv_client.push(name=config.name, ids=all_keys, data=initial_weights.cpu())
    
    print(f"\nSetting up optimizers with LR = {LEARNING_RATE}.")
    standard_optimizer = torch.optim.SGD(standard_ebc.parameters(), lr=LEARNING_RATE)
    sparse_optimizer = SparseSGD([recstore_ebc], lr=LEARNING_RATE)

    all_rounds_ok = True
    
    for i in range(NUM_TEST_ROUNDS):
        print("\n" + "#"*50)
        print(f"### Starting Test Round {i + 1} of {NUM_TEST_ROUNDS} ###")
        print("#"*50)

        batch = generate_random_batch(args.num_embeddings, args.batch_size, device)
        print(f"Generated a new random batch with {batch.values().numel()} values.")

        standard_output_kt = standard_ebc(batch)
        recstore_output_kt = recstore_ebc(batch)
        
        forward_pass_ok = compare_tensors(
            standard_output_kt.values(), 
            recstore_output_kt.values(), 
            f"Round {i+1} Forward Pass"
        )
        if not forward_pass_ok:
            print(f"🔥🔥🔥 Halting test: Forward pass failed in round {i+1}.")
            all_rounds_ok = False
            break

        dummy_loss_standard = standard_output_kt.values().sum()
        dummy_loss_recstore = recstore_output_kt.values().sum()
        
        standard_optimizer.zero_grad()
        sparse_optimizer.zero_grad()

        dummy_loss_standard.backward()
        dummy_loss_recstore.backward()

        standard_optimizer.step()
        sparse_optimizer.step()

        with torch.no_grad():
            config = eb_configs[0]
            updated_standard_weights = standard_ebc.state_dict()[f"embedding_bags.{config.name}.weight"]
            all_keys = torch.arange(config.num_embeddings, dtype=torch.int64)
            updated_recstore_weights = kv_client.pull(name=config.name, ids=all_keys).to(device)
            
            weights_ok = compare_tensors(
                updated_standard_weights, 
                updated_recstore_weights, 
                f"Round {i+1} Updated Weights"
            )
            if not weights_ok:
                print(f"🔥🔥🔥 Halting test: Weight update check failed in round {i+1}.")
                all_rounds_ok = False
                break
        
        print(f"--- Round {i+1} completed successfully ---")

    print("\n" + "="*30)
    print("### FINAL TEST SUMMARY ###")
    print("="*30)
    if all_rounds_ok:
        print(f"🎉🎉🎉 All {NUM_TEST_ROUNDS} precision test rounds passed! Your implementation is robust.")
    else:
        print("🔥🔥🔥 One or more precision test rounds failed. Please review the logs above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-round precision test for RecStore EBC.")
    parser.add_argument("--num-embeddings", type=int, default=1000, help="Number of embeddings per table.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Dimension of embeddings.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generated data.")
    parser.add_argument("--seed", type=int, default=int(torch.rand(1)[0]), help="Random seed for reproducibility.")
    parser.add_argument("--cpu", action="store_true", help="Force test to run on CPU.")
    
    args = parser.parse_args()
    main(args)
