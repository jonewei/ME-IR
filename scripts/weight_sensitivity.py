import json
import numpy as np

# 1. 配置路径
SEMANTIC_RAW = "results/raw_sem_scores.json" 
STRUCTURAL_RAW = "results/raw_str_scores.json"
QRELS_FILE = "data/qrel_76_expert.json"

def get_ranks_from_scores(raw_data_path):
    """将原始分值文件转换为排序后的 ID 列表"""
    with open(raw_data_path, 'r') as f:
        data = json.load(f)
    
    ranked_results = {}
    for qid, doc_scores in data.items():
        # 假设 doc_scores 是 {doc_id: score, ...}
        # 按 score 降序排列，只取前 1000 个以保证计算效率
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        ranked_results[qid] = [doc[0] for doc in sorted_docs[:1000]]
    return ranked_results

def calculate_mrr(results, qrels):
    mrr_list = []
    for qid, pred_ids in results.items():
        if qid not in qrels: continue
        gold_ids = set(qrels[qid])
        rank = 0
        for i, pid in enumerate(pred_ids):
            if pid in gold_ids:
                rank = i + 1
                break
        mrr_list.append(1.0 / rank if rank > 0 else 0)
    return np.mean(mrr_list)

def run_weight_test():
    print("🚀 正在预处理原始分值并生成排名列表...")
    sem_ranks = get_ranks_from_scores(SEMANTIC_RAW)
    str_ranks = get_ranks_from_scores(STRUCTURAL_RAW)
    
    with open(QRELS_FILE, 'r') as f: 
        qrels = json.load(f)

    k = 60
    weights = np.arange(0.1, 1.0, 0.1) 
    
    print("\n| Structural Weight ($w_{str}$) | MRR (76 Queries) | Performance Note |")
    print("|-------------------------------|------------------|------------------|")

    results_for_plot = []

    for w_str in weights:
        hybrid_results = {}
        for qid in sem_ranks.keys():
            rrf_scores = {}
            # 1. 处理语义流排名
            for rank, doc_id in enumerate(sem_ranks[qid]):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            
            # 2. 处理结构流排名 (加权融合)
            if qid in str_ranks:
                for rank, doc_id in enumerate(str_ranks[qid]):
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + w_str * (1.0 / (k + rank + 1))
            
            # 3. 最终混合重排序
            final_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            hybrid_results[qid] = [x[0] for x in final_ranked[:10]] # 只取前10看 MRR

        current_mrr = calculate_mrr(hybrid_results, qrels)
        
        note = "★ Optimal" if abs(w_str - 0.3) < 0.05 else ""
        print(f"| {w_str:<29.1f} | {current_mrr:<16.4f} | {note:<16} |")
        results_for_plot.append((w_str, current_mrr))

    # 输出用于绘图的 Python 数组，方便下一步直接画图
    print("\n📊 绘图数据备份 (Copy this to plot):")
    print(f"w_values = {[round(x[0], 1) for x in results_for_plot]}")
    print(f"mrr_values = {[round(x[1], 4) for x in results_for_plot]}")

if __name__ == "__main__":
    run_weight_test()