
# import json
# import os
# import numpy as np
# import time
# import sys
# from tqdm import tqdm
# from collections import defaultdict

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from retrieval.path_inverted_index import PathInvertedIndex
# from evaluation.final_hybrid_evaluator import HybridEvaluator

# # --- 配置区 ---
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# CORPUS_PATH = os.path.join(PROJECT_ROOT, "data/processed/formulas.json")
# QUERY_PATH = os.path.join(PROJECT_ROOT, "data/processed/queries_full.json")
# RELEVANCE_PATH = os.path.join(PROJECT_ROOT, "data/processed/relevance_labels.json")
# INDEX_SAVE_PATH = os.path.join(PROJECT_ROOT, "artifacts/substructure_index.pkl")

# def calculate_expert_metrics(results, gt_ids, k_max=20):
#     """
#     计算 P@K, MAP, Bpref, nDCG 等全套学术指标
#     """
#     metrics = {}
#     k_list = [1, 5, 10, 15, 20]
    
#     # 查找排名
#     first_rank = None
#     for i, item in enumerate(results[:k_max]):
#         fid = str(item[0]) if isinstance(item, (tuple, list)) else str(item)
#         if fid in gt_ids:
#             first_rank = i + 1
#             break
            
#     # 初始化
#     for k in k_list: metrics[f"P@{k}"] = 0.0
#     metrics["MAP"] = 0.0
#     metrics["MRR"] = 0.0
#     metrics["nDCG"] = 0.0
#     metrics["Bpref"] = 0.0

#     if first_rank:
#         # 1. P@K
#         for k in k_list:
#             if first_rank <= k:
#                 metrics[f"P@{k}"] = 1.0 / k
        
#         # 2. MRR & MAP (单真值时两者相等)
#         metrics["MRR"] = 1.0 / first_rank
#         metrics["MAP"] = 1.0 / first_rank
        
#         # 3. nDCG (取 Top 20)
#         metrics["nDCG"] = 1.0 / np.log2(first_rank + 1)
        
#         # 4. Bpref (简化版: 1 - (相关项之前的非相关项数 / 总召回数))
#         # 在单真值检索中，Bpref = 1 - (rank-1)/k_max
#         metrics["Bpref"] = max(0, 1 - (first_rank - 1) / k_max)
        
#     return metrics

# def reciprocal_rank_fusion(results_list, k=60, top_n=100):
#     rrf_scores = defaultdict(float)
#     for results in results_list:
#         if not results: continue
#         for rank, item in enumerate(results, start=1):
#             fid = str(item[0]) if isinstance(item, (tuple, list)) else str(item)
#             rrf_scores[fid] += 1.0 / (k + rank)
#     return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

# def run_experiment():
#     print("🧪 启动学术级多维评估程序...")
#     with open(CORPUS_PATH, 'r') as f: corpus = json.load(f)
#     with open(QUERY_PATH, 'r') as f: queries = json.load(f)
#     with open(RELEVANCE_PATH, 'r') as f: relevance = json.load(f)
#     sub_index = PathInvertedIndex.load(INDEX_SAVE_PATH)
#     vector_searcher = HybridEvaluator()

#     test_qids = [qid for qid in queries.keys() if qid in relevance]
#     results_storage = defaultdict(lambda: defaultdict(list))

#     for qid in tqdm(test_qids):
#         q_latex = queries[qid]  
#         gt_ids = set(str(k) for k in relevance[qid].keys())

#         v_res = vector_searcher.search_single(q_latex)[:100]
#         s_res = sub_index.search(q_latex, top_k=100)
#         h_res = reciprocal_rank_fusion([v_res, s_res], k=60, top_n=100)

#         for name, res in [("Vector", v_res), ("Substructure", s_res), ("Hybrid", h_res)]:
#             m = calculate_expert_metrics(res, gt_ids)
#             for metric_name, val in m.items():
#                 results_storage[name][metric_name].append(val)

#     # --- 打印精美学术报表 ---
#     print("\n" + "═"*110)
#     header = f"{'Method':<12} | {'P@1':<6} | {'P@5':<6} | {'P@10':<6} | {'P@20':<6} | {'MAP':<6} | {'MRR':<6} | {'nDCG':<6} | {'Bpref':<6}"
#     print(header)
#     print("─"*110)
#     for name in ["Vector", "Substructure", "Hybrid"]:
#         d = results_storage[name]
#         row = (f"{name:<12} | "
#                f"{np.mean(d['P@1']):.3f} | {np.mean(d['P@5']):.3f} | {np.mean(d['P@10']):.3f} | {np.mean(d['P@20']):.3f} | "
#                f"{np.mean(d['MAP']):.3f} | {np.mean(d['MRR']):.3f} | {np.mean(d['nDCG']):.3f} | {np.mean(d['Bpref']):.3f}")
#         print(row)
#     print("═"*110)

# if __name__ == "__main__":
#     run_experiment()

import json
import numpy as np
import pandas as pd
from collections import defaultdict

class FinalHybridEvaluator:
    def __init__(self, qrel_path, sem_path, str_path):
        with open(qrel_path, 'r') as f: self.qrels = json.load(f)
        with open(sem_path, 'r') as f: self.sem_run = json.load(f)
        with open(str_path, 'r') as f: self.str_run = json.load(f)
        self.k_rrf = 60
        self.w_str = 0.3  # 经过验证的最佳结构流权重

    def _calculate_metrics(self, run_results):
        metrics = defaultdict(list)
        
        for qid, target_docs in self.qrels.items():
            if qid not in run_results: continue
            
            # 排序检索结果 (Top 1000)
            retrieved = sorted(run_results[qid].items(), key=lambda x: x[1], reverse=True)[:1000]
            retrieved_ids = [str(doc_id) for doc_id, _ in retrieved]
            
            # 获取相关文档及其相关度
            rel_docs = {str(k): v for k, v in target_docs.items() if v > 0}
            non_rel_docs = {str(k) for k, v in target_docs.items() if v == 0}
            
            if not rel_docs: continue
            R = len(rel_docs)

            # 1. Precision @ K
            for k in [1, 5, 10, 20]:
                hits = len([d for d in retrieved_ids[:k] if d in rel_docs])
                metrics[f"P@{k}"].append(hits / k)

            # 2. MRR
            mrr = 0
            for i, d in enumerate(retrieved_ids):
                if d in rel_docs:
                    mrr = 1.0 / (i + 1)
                    break
            metrics["MRR"].append(mrr)

            # 3. MAP
            ap, hits = 0, 0
            for i, d in enumerate(retrieved_ids):
                if d in rel_docs:
                    hits += 1
                    ap += hits / (i + 1)
            metrics["MAP"].append(ap / R)

            # 4. nDCG (at 20)
            dcg = 0
            for i, d in enumerate(retrieved_ids[:20]):
                if d in rel_docs:
                    dcg += rel_docs[d] / np.log2(i + 2)
            idcg = sum([v / np.log2(i + 2) for i, v in enumerate(sorted(rel_docs.values(), reverse=True)[:20])])
            metrics["nDCG"].append(dcg / idcg if idcg > 0 else 0)

            # 5. Bpref
            # Bpref = 1/R * sum_{r in rel} (1 - #non-rel ranked above r / min(R, #non-rel-judged))
            if non_rel_docs:
                bpref_sum = 0
                relevant_found = [d for d in retrieved_ids if d in rel_docs]
                for r_doc in relevant_found:
                    rank_r = retrieved_ids.index(r_doc)
                    non_rel_above = len([d for d in retrieved_ids[:rank_r] if d in non_rel_docs])
                    bpref_sum += (1 - non_rel_above / min(R, len(non_rel_docs)))
                metrics["Bpref"].append(max(0, bpref_sum / R))
            else:
                metrics["Bpref"].append(mrr) # 回退方案

        return {k: np.mean(v) for k, v in metrics.items()}

    def hybrid_fuse(self):
        fused = defaultdict(dict)
        qids = set(self.sem_run.keys()) | set(self.str_run.keys())
        for qid in qids:
            scores = defaultdict(float)
            if qid in self.sem_run:
                for r, (d, _) in enumerate(sorted(self.sem_run[qid].items(), key=lambda x: x[1], reverse=True)):
                    scores[d] += 1.0 / (self.k_rrf + r + 1)
            if qid in self.str_run:
                for r, (d, _) in enumerate(sorted(self.str_run[qid].items(), key=lambda x: x[1], reverse=True)):
                    scores[d] += self.w_str / (self.k_rrf + r + 1)
            fused[qid] = dict(scores)
        return fused

    def print_table(self):
        print("Evaluating all methods...")
        res_vec = self._calculate_metrics(self.sem_run)
        res_str = self._calculate_metrics(self.str_run)
        res_hyb = self._calculate_metrics(self.hybrid_fuse())

        order = ["P@1", "P@5", "P@10", "P@20", "MAP", "MRR", "nDCG", "Bpref"]
        
        header = "Method       | " + " | ".join([f"{m:<6}" for m in order])
        line = "─" * len(header)
        double_line = "═" * len(header)

        def format_row(name, res_dict):
            vals = " | ".join([f"{res_dict[m]:.3f}" for m in order])
            return f"{name:<12} | {vals}"

        print(double_line)
        print(header)
        print(line)
        print(format_row("Vector", res_vec))
        print(format_row("Substructure", res_str))
        print(format_row("Hybrid", res_hyb))
        print(double_line)

if __name__ == "__main__":
    evaluator = FinalHybridEvaluator(
        "data/qrel_76_expert.json", 
        "results/raw_sem_scores.json", 
        "results/raw_str_scores.json"
    )
    evaluator.print_table()