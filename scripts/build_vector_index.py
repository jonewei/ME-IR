import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import os

# --- 配置 ---
MODEL_NAME = "witiko/mathberta"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256  # 3090 显存大，可以设高提升吞吐
CHECKPOINT_STEP = 50000  # 每 5 万条公式保存一次断点
ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "vector_index.faiss"
MAPPING_PATH = ARTIFACTS_DIR / "vector_id_mapping.json"
STATE_PATH = ARTIFACTS_DIR / "build_state.json"

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
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings

def load_state():
    """读取断点信息"""
    if STATE_PATH.exists():
        with open(STATE_PATH, 'r') as f:
            return json.load(f)
    return {"last_processed_idx": 0}

def save_state(idx):
    """保存断点信息"""
    with open(STATE_PATH, 'w') as f:
        json.dump({"last_processed_idx": idx}, f)

def build_index():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    # 1. 加载数据
    print("📖 正在读取 formulas.json ...")
    with open("data/processed/formulas.json", 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    fids = list(corpus.keys())
    # 2. 这里的内存管理：只保留当前需要的列表，尽快释放 corpus
    latex_list = [corpus[fid]['latex_norm'] for fid in fids]
    del corpus # 释放大字典，腾出内存给向量
    
    # 3. 初始化或恢复索引
    state = load_state()
    start_idx = state["last_processed_idx"]
    dimension = 768
    
    if start_idx > 0 and INDEX_PATH.exists():
        print(f"🔄 检测到断点，准备从第 {start_idx:,} 条公式继续...")
        index = faiss.read_index(str(INDEX_PATH))
        with open(MAPPING_PATH, 'r') as f:
            saved_fids = json.load(f)
    else:
        print("🏗️ 初始化全新索引...")
        # 针对 30GB 内存，如果全量跑 1300 万，后期建议换成 IndexIVFPQ (压缩索引)
        # 目前 5 分片测试，IndexFlatIP 完全没问题
        index = faiss.IndexFlatIP(dimension)
        saved_fids = []
        start_idx = 0

    engine = MathVectorEngine()
    
    # 4. 循环编码
    print(f"🚀 开始向量化 (目标: {len(latex_list):,} 条)...")
    pbar = tqdm(total=len(latex_list), initial=start_idx, desc="向量编码")
    
    for i in range(start_idx, len(latex_list), BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, len(latex_list))
        batch = latex_list[i : end_idx]
        batch_fids = fids[i : end_idx]
        
        try:
            emb = engine.encode(batch)
            faiss.normalize_L2(emb)
            index.add(emb)
            saved_fids.extend(batch_fids)
            
            # 定期保存断点，防止崩溃
            if (i + BATCH_SIZE) % CHECKPOINT_STEP == 0 or end_idx == len(latex_list):
                faiss.write_index(index, str(INDEX_PATH))
                with open(MAPPING_PATH, 'w') as f:
                    json.dump(saved_fids, f)
                save_state(end_idx)
            
            pbar.update(len(batch))
        except Exception as e:
            print(f"\n❌ 出错于索引 {i}: {e}")
            continue
            
    pbar.close()
    print(f"✅ 完成！总索引数: {index.ntotal:,}")

if __name__ == "__main__":
    build_index()