import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from retrieval.approach0_hash import DualHashGenerator, Approach0HashIndex

# 配置路径
FORMULAS_JSON = "data/processed/formulas.json"
QUERIES_JSON = "data/processed/queries_full.json"
LABELS_JSON = "data/processed/relevance_labels.json"

def analyze_errors():
    print("🔍 启动错误分析系统 (Recall=0 案例提取)...")
    
    # 1. 加载资源
    with open(FORMULAS_JSON, 'r') as f:
        corpus = json.load(f)
    with open(QUERIES_JSON, 'r') as f:
        queries = json.load(f)
    with open(LABELS_JSON, 'r') as f:
        relevance = json.load(f)
    
    # 假设你已经运行过评测，这里我们重新运行逻辑寻找失败者
    # (为了简化，这里直接对比 relevance 中的 ID 是否在 Top-1000 逻辑外)
    
    # 我们假设使用你之前的 Hybrid Evaluator 逻辑进行模拟
    from evaluation.final_hybrid_evaluator import HybridEvaluator
    evaluator = HybridEvaluator()
    
    failed_cases = []

    print("🧪 正在扫描失败查询...")
    for qid, query_latex in queries.items():
        gt_dict = relevance.get(qid, {})
        if not gt_dict: continue
        
        gt_ids = set(str(vid) for vid in gt_dict.keys())
        results = evaluator.search_single(query_latex)
        
        hits = gt_ids.intersection(set(results))
        if len(hits) == 0:
            failed_cases.append({
                "qid": qid,
                "query": query_latex,
                "gt_sample_ids": list(gt_ids)[:3] # 取前3个答案做对比
            })

    # 2. 输出分析报告
    report_path = "evaluation/error_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"📊 错误分析报告 (Total Failed: {len(failed_cases)})\n")
        f.write("="*80 + "\n\n")
        
        for case in failed_cases:
            f.write(f"❌ Topic ID: {case['qid']}\n")
            f.write(f"   [Query LaTeX]: {case['query']}\n")
            f.write(f"   [Ground Truths]:\n")
            
            for g_id in case['gt_sample_ids']:
                if g_id in corpus:
                    gt_latex = corpus[g_id]['latex']
                    f.write(f"      - ID {g_id}: {gt_latex}\n")
                else:
                    f.write(f"      - ID {g_id}: ⚠️ 库中不存在 (Coverage Error)\n")
            f.write("-" * 80 + "\n")

    print(f"✅ 分析完成！报告已保存至: {report_path}")
    print(f"💡 提示：请打开该文件，肉眼比对 Query 和 GT 的 LaTeX 写法差异。")

if __name__ == "__main__":
    analyze_errors()