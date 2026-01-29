import json
import os
import torch
import numpy as np
from sentence_transformers import CrossEncoder

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts/cross_encoder_model")
CORPUS_PATH = os.path.join(PROJECT_ROOT, "data/processed/formulas.json")
QUERY_PATH = os.path.join(PROJECT_ROOT, "data/processed/queries_full.json")
RELEVANCE_PATH = os.path.join(PROJECT_ROOT, "data/processed/relevance_labels.json")

def check():
    print("🔍 启动深度逻辑诊断...")
    model = CrossEncoder(MODEL_PATH, device="cuda")
    
    with open(CORPUS_PATH, 'r') as f: corpus = json.load(f)
    with open(QUERY_PATH, 'r') as f: queries = json.load(f)
    with open(RELEVANCE_PATH, 'r') as f: relevance = json.load(f)

    # 1. 挑选一个测试 Query
    qid = list(relevance.keys())[0]
    q_latex = queries[qid]
    gt_ids = list(relevance[qid].keys())
    
    print(f"\n❓ Query ID: {qid}")
    print(f"❓ Query 内容: {q_latex}")
    print("-" * 50)

    # 2. 准备正样本和负样本
    pos_id = gt_ids[0]
    neg_id = list(corpus.keys())[100] # 随便找个大概率不相关的
    
    samples = [
        ("✅ 正样本 (GT)", pos_id),
        ("❌ 负样本 (Random)", neg_id)
    ]

    for label, rid in samples:
        doc = corpus[rid]
        raw_latex = doc.get('latex', '')
        norm_latex = doc.get('latex_norm', '')

        # 推理
        s_raw = model.predict([q_latex, raw_latex])
        s_norm = model.predict([q_latex, norm_latex])

        print(f"{label} [ID: {rid}]:")
        print(f"  - 原始 LaTeX 得分: {s_raw:.4f}")
        print(f"  - 规范化 LaTeX 得分: {s_norm:.4f}")
        print(f"  - 内容预览: {raw_latex[:60]}...")
        print("-" * 20)

if __name__ == "__main__":
    check()