import pandas as pd
import numpy as np
import json
import time
import scipy.stats as stats
from collections import defaultdict
import os

# 核心评估库：pip install pytrec_eval
try:
    import pytrec_eval
except ImportError:
    print("Error: pytrec_eval is required. Install with: pip install pytrec_eval")

class LS_MIR_ExperimentRunner:
    def __init__(self, qrel_path, str_results_path, sem_results_path, query_metadata_path=None):
        """
        LS-MIR 实验自动化运行器
        :param qrel_path: 真值文件 (JSON: {qid: {docid: rel, ...}})
        :param str_results_path: IPI 结构流检索结果
        :param sem_results_path: 语义流检索结果 (如 Math-BERT)
        :param query_metadata_path: 包含 query 长度等信息的文件，用于复杂度分析
        """
        print(f"[{time.strftime('%H:%M:%S')}] 加载实验数据...")
        
        with open(qrel_path, 'r') as f:
            self.qrel = json.load(f)
        with open(str_results_path, 'r') as f:
            self.str_run = json.load(f)
        with open(sem_results_path, 'r') as f:
            self.sem_run = json.load(f)
            
        self.query_metadata = {}
        if query_metadata_path and os.path.exists(query_metadata_path):
            with open(query_metadata_path, 'r') as f:
                self.query_metadata = json.load(f) # {qid: {"latex": "...", "length": 45}}

        # 初始化评估器
        self.evaluator = pytrec_eval.RelevanceEvaluator(
            self.qrel, {'recip_rank', 'ndcg_cut_10', 'map', 'P_1'}
        )
        self.lsmir_run_cache = None

    # --- 核心融合算法 ---
    def reciprocal_rank_fusion(self, run_str, run_sem, k=60):
        """实现 RRF 排名共识算法"""
        start_time = time.perf_counter()
        fusion_run = defaultdict(dict)
        qids = run_str.keys()
        
        for qid in qids:
            # 获取两个流的排名
            docs_str = sorted(run_str[qid].items(), key=lambda x: x[1], reverse=True)
            docs_sem = sorted(run_sem[qid].items(), key=lambda x: x[1], reverse=True)
            
            # 计算得分
            for rank, (doc_id, _) in enumerate(docs_str):
                fusion_run[qid][doc_id] = fusion_run[qid].get(doc_id, 0) + 1.0 / (k + rank + 1)
            for rank, (doc_id, _) in enumerate(docs_sem):
                fusion_run[qid][doc_id] = fusion_run[qid].get(doc_id, 0) + 1.0 / (k + rank + 1)
        
        latency = (time.perf_counter() - start_time) / len(qids) * 1000 # 毫秒
        return fusion_run, latency

    def linear_fusion(self, run_str, run_sem, alpha=0.5):
        """线性得分融合 (用于消融实验对比)"""
        fusion_run = defaultdict(dict)
        for qid in run_str.keys():
            all_docs = set(run_str[qid].keys()) | set(run_sem[qid].keys())
            for doc_id in all_docs:
                s_str = run_str[qid].get(doc_id, 0)
                s_sem = run_sem[qid].get(doc_id, 0)
                fusion_run[qid][doc_id] = alpha * s_str + (1 - alpha) * s_sem
        return fusion_run

    # --- 实验模块 ---
    def run_ablation_study(self):
        """4.3 消融实验：验证 IPI, Semantic 和 RRF 的各自贡献"""
        print("\n>>> 执行消融实验 (Ablation Study)...")
        results = []
        
        # S1: 纯语义
        res_s1 = self.evaluator.evaluate(self.sem_run)
        results.append({"Setting": "S1: Semantic only", **self._avg_metrics(res_s1)})
        
        # S2: 纯结构
        res_s2 = self.evaluator.evaluate(self.str_run)
        results.append({"Setting": "S2: Structural only (IPI)", **self._avg_metrics(res_s2)})
        
        # S3: 线性混合 (未通过RRF排名共识)
        lin_run = self.linear_fusion(self.str_run, self.sem_run, alpha=0.5)
        res_s3 = self.evaluator.evaluate(lin_run)
        results.append({"Setting": "S3: Linear Mix (0.5/0.5)", **self._avg_metrics(res_s3)})
        
        # S4: LS-MIR (RRF 排名共识)
        self.lsmir_run_cache, _ = self.reciprocal_rank_fusion(self.str_run, self.sem_run, k=60)
        res_s4 = self.evaluator.evaluate(self.lsmir_run_cache)
        results.append({"Setting": "S4: LS-MIR (Proposed)", **self._avg_metrics(res_s4)})
        
        df = pd.DataFrame(results)
        return df

    def run_parameter_sensitivity(self):
        """4.4 参数敏感性：RRF 中 k 值的变化"""
        print("\n>>> 执行参数敏感性分析 (k-value)...")
        k_results = []
        for k in [10, 20, 40, 60, 80, 100]:
            run, _ = self.reciprocal_rank_fusion(self.str_run, self.sem_run, k=k)
            metrics = self._avg_metrics(self.evaluator.evaluate(run))
            k_results.append({"k": k, "MRR": metrics['MRR'], "nDCG@10": metrics['nDCG@10']})
        return pd.DataFrame(k_results)

    def run_advanced_analysis(self):
        """显著性检验、复杂度分析与效率统计"""
        print("\n>>> 执行深度分析 (Significance & Complexity)...")
        
        # 1. 显著性检验 (Paired t-test)
        lsmir_scores = [m['recip_rank'] for m in self.evaluator.evaluate(self.lsmir_run_cache).values()]
        bert_scores = [m['recip_rank'] for m in self.evaluator.evaluate(self.sem_run).values()]
        t_stat, p_val = stats.ttest_rel(lsmir_scores, bert_scores)
        
        # 2. 复杂度分解
        complexity_res = []
        if self.query_metadata:
            groups = {'Simple (<20)': [], 'Medium (20-50)': [], 'Complex (>50)': []}
            all_res = self.evaluator.evaluate(self.lsmir_run_cache)
            for qid, m in all_res.items():
                length = self.query_metadata.get(qid, {}).get('length', 0)
                if length < 20: groups['Simple (<20)'].append(m['recip_rank'])
                elif length <= 50: groups['Medium (20-50)'].append(m['recip_rank'])
                else: groups['Complex (>50)'].append(m['recip_rank'])
            
            for g, scores in groups.items():
                complexity_res.append({"Category": g, "Count": len(scores), "MRR": round(np.mean(scores), 4)})

        # 3. 效率统计
        _, rrf_latency = self.reciprocal_rank_fusion(self.str_run, self.sem_run, k=60)
        
        return p_val, pd.DataFrame(complexity_res), rrf_latency

    def _avg_metrics(self, res):
        """计算平均指标"""
        metrics = {'P@1': 'P_1', 'MRR': 'recip_rank', 'nDCG@10': 'ndcg_cut_10', 'MAP': 'map'}
        out = {}
        for label, key in metrics.items():
            out[label] = round(np.mean([m[key] for m in res.values()]), 4)
        return out

# --- 执行主程序 ---
if __name__ == "__main__":
    # 路径配置 (请确保文件存在于对应路径)
    runner = LS_MIR_ExperimentRunner(
        qrel_path='data/qrel_76_expert.json',
        str_results_path='results/raw_str_scores.json', 
        sem_results_path='results/raw_sem_scores.json',
        query_metadata_path='data/query_metadata.json'
    )
    
    # 1. 消融实验
    ablation_df = runner.run_ablation_study()
    print("\n[Table 4.2 Ablation Study Results]")
    print(ablation_df.to_markdown(index=False))
    
    # 2. 参数调优
    k_df = runner.run_parameter_sensitivity()
    print("\n[Table 4.3 Parameter Sensitivity (k)]")
    print(k_df.to_markdown(index=False))
    
    # 3. 深度分析
    p_val, complexity_df, rrf_latency = runner.run_advanced_analysis()
    
    print(f"\n[Statistical Significance]")
    print(f"Paired T-test (LS-MIR vs Math-BERT) p-value: {p_val:.6e}")
    if p_val < 0.01: print("Result: Highly Significant (p < 0.01)")
    
    if not complexity_df.empty:
        print("\n[Table 5.x Performance by Complexity]")
        print(complexity_df.to_markdown(index=False))
    
    print(f"\n[Efficiency Analysis]")
    print(f"Average Rank Fusion Latency: {rrf_latency:.2f} ms")