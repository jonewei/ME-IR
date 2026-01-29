import json
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine

# --- 配置区 ---
MODEL_NAME = "math-similarity/Bert-MLM_arXiv-MP-class_zbMath"
RELEVANCE_PATH = "data/processed/relevance_labels.json"
QUERY_PATH = "data/processed/queries_full.json"
CORPUS_PATH = "data/processed/formulas.json"

# --- 1. 加载模型（数学专用版） ---
print(f"📡 正在加载数学专家模型: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 取 [CLS] 向量作为表征
    return outputs.last_hidden_state[0][0].numpy()

# --- 2. 加载实验数据 ---
with open(RELEVANCE_PATH, 'r') as f: relevance = json.load(f)
with open(QUERY_PATH, 'r') as f: queries = json.load(f)
with open(CORPUS_PATH, 'r') as f: corpus = json.load(f)

# --- 3. 核心实验：针对 76 条 Query 进行语义敏感度测试 ---
print("🧪 开始数学语义对标测试 (Sampled Reranking)...")

results = []
# 为了快速验证，我们选取你之前评估过的 test_qids
for qid in tqdm(list(relevance.keys())[:20]):  # 先测20条看趋势
    q_latex = queries[qid]
    gt_id = list(relevance[qid].keys())[0]  # 获取真值ID
    gt_latex = corpus[str(gt_id)]['latex_norm']
    
    # 获取数学 BERT 的编码
    q_vec = get_embedding(q_latex)
    gt_vec = get_embedding(gt_latex)
    
    # 计算余弦相似度
    sim_score = 1 - cosine(q_vec, gt_vec)
    results.append(sim_score)

print(f"\n✅ 实验完成！")
print(f"📊 Math-BERT 对真值公式的平均语义相似度 (Similarity Score): {np.mean(results):.4f}")