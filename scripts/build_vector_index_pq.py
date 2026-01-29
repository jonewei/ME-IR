import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import json
import os

# --- 配置 ---
MODEL_NAME = "witiko/mathberta"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128  # 3090 建议设为 128 或 256
ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "vector_index_pq.faiss"
MAPPING_PATH = ARTIFACTS_DIR / "vector_id_mapping_pq.json"

class MathVectorEngine:
    def __init__(self):
        print(f"正在加载模型 {MODEL_NAME} 到 {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()

    def encode(self, latex_list):
        inputs = self.tokenizer(latex_list, padding=True, truncation=True, 
                                 max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 取 CLS 向量
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings

def build_index():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    # 1. 加载数据 ID
    print("📖 正在读取 formulas.json ...")
    with open("data/processed/formulas.json", 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    # 修改这里：强制只取前 10 万条（大约相当于 1 个分片的量）进行测试
    all_fids = list(corpus.keys())
    fids = list(corpus.keys())
    # fids = all_fids[:1000000] 
    print(f"🧪 测试模式：仅处理前 {len(fids)} 条公式")
    
    # 
    
    engine = MathVectorEngine()
    dimension = 768
    
    # 2. 初始化或加载索引
    if not INDEX_PATH.exists():
        print("🏗️ 正在准备训练 PQ 压缩索引...")
        quantizer = faiss.IndexFlatIP(dimension)
        # m=96 (768/8), nlist=1024
        index = faiss.IndexIVFFlat(quantizer, dimension, 1024, faiss.METRIC_INNER_PRODUCT)
        
        # --- 核心修复：分批提取训练数据 ---
        train_size = min(100000, len(fids))
        print(f"🧪 正在编码 {train_size} 条数据用于训练索引...")
        train_embs = []
        for i in tqdm(range(0, train_size, BATCH_SIZE), desc="训练数据编码"):
            batch_fids = fids[i : i + BATCH_SIZE]
            batch_latex = [corpus[fid]['latex_norm'] for fid in batch_fids]
            emb = engine.encode(batch_latex)
            faiss.normalize_L2(emb)
            train_embs.append(emb)
        
        train_data = np.vstack(train_embs)
        print("⚙️ 正在训练聚类中心 (此步仅需 CPU/GPU 片刻)...")
        index.train(train_data)
        del train_data
        del train_embs
        saved_fids = []
    else:
        print("🔄 加载现有索引以继续...")
        index = faiss.read_index(str(INDEX_PATH))
        with open(MAPPING_PATH, 'r') as f:
            saved_fids = json.load(f)

    # 3. 循环编码与添加
    print(f"🚀 开始向量化 (剩余: {len(fids) - index.ntotal:,} 条)...")
    start_idx = index.ntotal
    
    pbar = tqdm(total=len(fids), initial=start_idx, desc="PQ 编码中")
    
    for i in range(start_idx, len(fids), BATCH_SIZE):
        batch_fids = fids[i : i + BATCH_SIZE]
        batch_latex = [corpus[fid]['latex_norm'] for fid in batch_fids]
        
        try:
            emb = engine.encode(batch_latex)
            faiss.normalize_L2(emb)
            index.add(emb)
            saved_fids.extend(batch_fids)
            
            # 每 10 万条保存一次磁盘
            if len(saved_fids) % 100000 == 0:
                faiss.write_index(index, str(INDEX_PATH))
                with open(MAPPING_PATH, 'w') as f:
                    json.dump(saved_fids, f)
            
            pbar.update(len(batch_latex))
        except Exception as e:
            print(f"跳过批次 {i} 由于错误: {e}")
            continue
            
    # 最终保存
    faiss.write_index(index, str(INDEX_PATH))
    with open(MAPPING_PATH, 'w') as f:
        json.dump(saved_fids, f)
    print(f"✅ 完成！最终索引大小: {index.ntotal:,}")

if __name__ == "__main__":
    build_index()