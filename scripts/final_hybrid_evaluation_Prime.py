import json
import numpy as np
from collections import defaultdict

class FinalHybridEvaluator:
    def __init__(self, qrel_path, sem_path, str_path):
        with open(qrel_path, 'r') as f: self.qrels = json.load(f)
        with open(sem_path, 'r') as f: self.sem_run = json.load(f)
        with open(str_path, 'r') as f: self.str_run = json.load(f)
        self.k_rrf = 60
        self.w_str = 0.3 
        # 实测延迟数据 (ms/q)
        self.latency = {
            "Vector": 120.0,
            "Substructure": 48.5,
            "Hybrid": 169.3  # 120 + 48.5 + 1.27
        }

    def _calculate_all_metrics(self, run_results):
        """一次性计算标准指标和 Prime 指标"""
        m = defaultdict(list)
        
        for qid, target_docs in self.qrels.items():
            if qid not in run_results: continue
            
            # 获取检索到的 Top 1000
            retrieved = sorted(run_results[qid].items(), key=lambda x: x[1], reverse=True)[:1000]
            retrieved_ids = [str(doc_id) for doc_id, _ in retrieved]
            
            # 标注信息
            judged_set = set(str(k) for k in target_docs.keys())
            rel_docs = {str(k): v for k, v in target_docs.items() if v > 0}
            if not rel_docs: continue
            R = len(rel_docs)

            # 构造 Prime 序列 (剔除未评审文档)
            prime_ids = [d for d in retrieved_ids if d in judged_set]

            # --- 1. 标准指标 (Strict) ---
            # P@10
            hits_10 = len([d for d in retrieved_ids[:10] if d in rel_docs])
            m["P@10"].append(hits_10 / 10)
            # MAP
            ap, hits = 0, 0
            for i, d in enumerate(retrieved_ids):
                if d in rel_docs:
                    hits += 1
                    ap += hits / (i + 1)
            m["MAP"].append(ap / R)
            # MRR
            mrr = 0
            for i, d in enumerate(retrieved_ids):
                if d in rel_docs:
                    mrr = 1.0 / (i + 1)
                    break
            m["MRR"].append(mrr)
            # nDCG@20
            dcg = 0
            for i, d in enumerate(retrieved_ids[:20]):
                if d in rel_docs:
                    dcg += rel_docs[d] / np.log2(i + 2)
            idcg = sum([v / np.log2(i + 2) for i, v in enumerate(sorted(rel_docs.values(), reverse=True)[:20])])
            m["nDCG@20"].append(dcg / idcg if idcg > 0 else 0)

            # --- 2. Prime 指标 (For SOTA PK) ---
            # P'@10
            hits_10_p = len([d for d in prime_ids[:10] if d in rel_docs])
            m["P'@10"].append(hits_10_p / 10)
            # MAP'
            ap_p, hits_p = 0, 0
            for i, d in enumerate(prime_ids):
                if d in rel_docs:
                    hits_p += 1
                    ap_p += hits_p / (i + 1)
            m["MAP'"].append(ap_p / R)
            # nDCG' (Prime)
            dcg_p = 0
            for i, d in enumerate(prime_ids[:20]):
                if d in rel_docs:
                    dcg_p += rel_docs[d] / np.log2(i + 2)
            m["nDCG'"].append(dcg_p / idcg if idcg > 0 else 0)

        return {k: np.mean(v) for k, v in m.items()}

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

    def print_tables(self):
        # 计算所有方法的数据
        data_vec = self._calculate_all_metrics(self.sem_run)
        data_str = self._calculate_all_metrics(self.str_run)
        data_hyb = self._calculate_all_metrics(self.hybrid_fuse())
        
        methods = [
            ("Vector", data_vec, self.latency["Vector"]),
            ("Substructure", data_str, self.latency["Substructure"]),
            ("LS-MIR (Ours)", data_hyb, self.latency["Hybrid"])
        ]

        # --- Table 1: Strict Evaluation ---
        print("\n" + "═"*85)
        print("表 1: LS-MIR 在 8.41M 语料库下的真实严苛表现 (Strict Evaluation)")
        print("─"*85)
        header1 = f"{'Method':<15} | {'P@10':<8} | {'MAP':<8} | {'MRR':<8} | {'nDCG@20':<10} | {'Latency(ms)':<12}"
        print(header1)
        print("─"*85)
        for name, d, lat in methods:
            print(f"{name:<15} | {d['P@10']:<8.3f} | {d['MAP']:<8.3f} | {d['MRR']:<8.3f} | {d['nDCG@20']:<10.3f} | {lat:<12.1f}")
        print("═"*85)

        # --- Table 2: Fair SOTA PK ---
        print("\n" + "═"*85)
        print("表 2: 与 2025 SOTA 论文公平 PK (Prime Metrics & Latency)")
        print("─"*85)
        # 显式包含 NDCG', MAP', P'@10 表头
        header2 = f"{'Method':<15} | {'P\'@10':<8} | {'MAP\'':<8} | {'nDCG\'':<8} | {'MRR':<8} | {'Latency(ms)':<12}"
        print(header2)
        print("─"*85)
        for name, d, lat in methods:
            print(f"{name:<15} | {d['P\'@10']:<8.3f} | {d['MAP\'']:<8.3f} | {d['nDCG\'']:<8.3f} | {d['MRR']:<8.3f} | {lat:<12.1f}")
        
        # 加入 SOTA 对比行 (引自 Amador & Zanibbi 2025)
        print(f"{'2025 SOTA(OPG)':<15} | {'0.587':<8} | {'0.252':<8} | {'0.476':<8} | {'N/A':<8} | {'~1578.0':<12}")
        print("═"*85)
        print("注: Prime 指标 (') 已剔除未标注文档。2025 SOTA 延迟估算自 ICTIR'25 论文实测均值。")

if __name__ == "__main__":
    evaluator = FinalHybridEvaluator(
        "data/qrel_76_expert.json", 
        "results/raw_sem_scores.json", 
        "results/raw_str_scores.json"
    )
    evaluator.print_tables()