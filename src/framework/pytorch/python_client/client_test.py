import torch
from client import RecstoreClient
import numpy as np
import sys

if len(sys.argv) < 2:
    raise RuntimeError("This script requires the path to the compiled C++ library as an argument.")
library_path = sys.argv[1]
print(f"--- Loading C++ library from: {library_path} ---")

client = RecstoreClient(library_path=library_path)
embedding_dim = 128

print("\n--- Test 1: Write and Read Verification ---")
keys_to_write = torch.tensor([1001, 1002, 1003], dtype=torch.int64)
values_to_write = torch.randn(3, embedding_dim, dtype=torch.float32)

print(f"Writing embeddings for keys: {keys_to_write.tolist()}")
client.emb_write(keys_to_write, values_to_write)
print("Write call successful.")

print(f"Reading embeddings for keys: {keys_to_write.tolist()}")
read_values = client.emb_read(keys_to_write, embedding_dim)

assert read_values.shape == values_to_write.shape, "Shape mismatch after read"
assert torch.allclose(read_values, values_to_write), "Value mismatch after read"
print("Read successful. Written values verified.")


print("\n--- Test 2: Reading Unseen Keys ---")
keys_to_read = torch.tensor([9998, 9999], dtype=torch.int64)
print(f"Reading embeddings for unseen keys: {keys_to_read.tolist()}")
read_values_unseen = client.emb_read(keys_to_read, embedding_dim)

assert read_values_unseen.shape == (2, embedding_dim)
assert torch.all(read_values_unseen == 0), "Unseen keys should return zero vectors"
print("Read for unseen keys successful, returned zero vectors as expected.")


# print("\n--- Test 3: Update Operation ---")
# keys_to_update = torch.tensor([1001, 1002], dtype=torch.int64)
# grads_to_update = torch.ones(2, embedding_dim, dtype=torch.float32)

# print(f"Updating embeddings for keys: {keys_to_update.tolist()}")
# client.emb_update(keys_to_update, grads_to_update)
# print("Update call successful.")

# print("Reading updated keys to verify update...")
# values_after_update = client.emb_read(keys_to_update, embedding_dim)
# expected_values = values_to_write[:2] - (0.01 * grads_to_update)

# assert torch.allclose(values_after_update, expected_values), "Values not correctly updated"
# print("Update verified successfully.")
