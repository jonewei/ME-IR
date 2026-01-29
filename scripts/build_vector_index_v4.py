import json
import faiss
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath'
BATCH_SIZE = 512
CHUNK_SIZE = 500000 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_vector_index():
    print(f"🚀 启动向量索引构建...")
    with open("data/processed/formulas.json", 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    vids = list(corpus.keys())
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    index = faiss.IndexFlatIP(768)
    mapping = []

    for i in range(0, len(vids), CHUNK_SIZE):
        batch_vids = vids[i : i + CHUNK_SIZE]
        texts = [corpus[vid]['latex_norm'] for vid in batch_vids]
        
        print(f"📦 Processing chunk {i//CHUNK_SIZE + 1}...")
        embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, 
                                  normalize_embeddings=True, convert_to_numpy=True)
        index.add(embeddings.astype('float32'))
        mapping.extend(batch_vids)
        torch.cuda.empty_cache()

    faiss.write_index(index, "artifacts/vector_index_full_v4.faiss")
    with open("artifacts/vector_id_mapping_v4.json", 'w') as f:
        json.dump(mapping, f)
    print(f"✅ 构建完成，总向量数: {index.ntotal}")

if __name__ == "__main__":
    build_vector_index()