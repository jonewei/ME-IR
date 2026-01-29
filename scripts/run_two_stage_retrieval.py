import json
import torch
import numpy as np
import os
from tqdm import tqdm
from sentence_transformers import CrossEncoder

# 1. 自动获取路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts/cross_encoder_model")
RELEVANCE_PATH = os.path.join(PROJECT_ROOT, "data/processed/relevance_labels.json")
CORPUS_PATH = os.path.join(PROJECT_ROOT, "data/processed/formulas.json")
QUERY_PATH = os.path.join(PROJECT_ROOT, "data/processed/queries_full.json")

# 尝试导入评估器
try:
    from evaluation.final_hybrid_evaluator import HybridEvaluator
except ImportError:
    from final_hybrid_evaluator import HybridEvaluator

def extract_latex(item):
    """确保提取出的是字符串 LaTeX"""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("latex_norm") or item.get("latex") or ""
    return str(item) if item is not None else ""

def evaluate_two_stage():
    print(f"📦 正在初始化两阶段检索系统...")
    
    # 初始化粗排和精排
    hybrid_searcher = HybridEvaluator()
    reranker = CrossEncoder(MODEL_PATH, device="cuda")
    
    print("📖 加载索引与标签数据...")
    with open(RELEVANCE_PATH, 'r') as f: relevance = json.load(f)
    with open(QUERY_PATH, 'r') as f: queries = json.load(f)
    with open(CORPUS_PATH, 'r') as f: corpus = json.load(f)
    
    annotated_qids = [qid for qid in queries.keys() if qid in relevance]
    initial_mrrs, reranked_mrrs = [], []
    
    print(f"🚀 开始处理 {len(annotated_qids)} 个标注查询...")

    # 总进度条
    pbar = tqdm(annotated_qids, desc="Total Progress", unit="query")
    
    for qid in pbar:
        q_latex = extract_latex(queries[qid])
        gt_ids = set(str(k) for k in relevance[qid].keys())
        
        # --- 第一阶段：粗排 (召回 1000) ---
        initial_results = hybrid_searcher.search_single(q_latex)[:1000]
        if not initial_results:
            continue
            
        # 计算初始 MRR
        mrr_init = 0
        for i, res_id in enumerate(initial_results):
            if str(res_id) in gt_ids:
                mrr_init = 1 / (i + 1)
                break
        initial_mrrs.append(mrr_init)
        
        # --- 第二阶段：精排 (重排前 100) ---
        to_rerank_ids = initial_results[:100]
        # 核心修正：确保提取字符串而非字典
        candidates = [extract_latex(corpus.get(str(rid))) for rid in to_rerank_ids]
        
        # 过滤掉空字符串，防止模型报错
        valid_pairs = []
        valid_ids = []
        for rid, cand in zip(to_rerank_ids, candidates):
            if cand.strip():
                valid_pairs.append([q_latex, cand])
                valid_ids.append(rid)
        
        if valid_pairs:
            # 4090 极速精排推理
            scores = reranker.predict(valid_pairs, batch_size=128, show_progress_bar=False)
            
            # 按分数从高到低排序
            reranked_indices = np.argsort(scores)[::-1]
            # reranked_indices = np.argsort(scores)
            reranked_top_ids = [valid_ids[i] for i in reranked_indices]
            
            # 拼接结果：[精排后的有效ID] + [原始结果中未参与精排的部分]
            final_results = reranked_top_ids + [rid for rid in initial_results if rid not in valid_ids]
        else:
            final_results = initial_results

        # 计算精排后的 MRR
        mrr_rerank = 0
        for i, res_id in enumerate(final_results):
            if str(res_id) in gt_ids:
                mrr_rerank = 1 / (i + 1)
                break
        reranked_mrrs.append(mrr_rerank)
        
        # 动态更新进度条显示的平均 MRR
        current_mrr = np.mean(reranked_mrrs)
        pbar.set_postfix({"Avg_MRR": f"{current_mrr:.4f}"})

    # --- 输出最终报告 ---
    m1, m2 = np.mean(initial_mrrs), np.mean(reranked_mrrs)
    print("\n" + "═"*50)
    print(f"🏆 实验结果收割报告")
    print("═"*50)
    print(f"📊 初始粗排 MRR (Baseline):   {m1:.4f}")
    print(f"🔥 精排重排 MRR (Two-Stage): {m2:.4f}")
    print(f"📈 性能净提升 (Absolute):     {m2-m1:+.4f}")
    print(f"🚀 相对增益 (Relative):        {(m2-m1)/m1:.2%}")
    print("═"*50)

if __name__ == "__main__":
    evaluate_two_stage()