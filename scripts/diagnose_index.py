import pickle
import os

INDEX_PATH = "data/indices/ipi_index.bin"
SHARD_PATH = "data/arqmath3/latex_representation_v3/1.tsv"

print("--- 1. 抽查索引内部的 Key ---")
with open(INDEX_PATH, 'rb') as f:
    index = pickle.load(f)
keys = list(index.keys())
print(f"索引总词项数: {len(keys)}")
print(f"前 20 个 Key 示例: {keys[:20]}")

print("\n--- 2. 抽查原始分片的前 2 行 ---")
with open(SHARD_PATH, 'r', encoding='utf-8') as f:
    for i in range(2):
        line = f.readline()
        parts = line.strip().split('\t')
        print(f"行 {i+1} 分段数: {len(parts)}")
        for idx, p in enumerate(parts):
            print(f"  列 {idx}: {p[:50]}...")