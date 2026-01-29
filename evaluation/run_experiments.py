# import json
# import numpy as np
# import pandas as pd
# from collections import defaultdict
# from scipy import stats
# from tabulate import tabulate
# import re # 确保文件开头导入了 re

# class Evaluator:
#     def __init__(self, qrel_path, sem_path, str_path):
#         self.qrel_path = qrel_path
#         self.sem_path = sem_path
#         self.str_path = str_path
#         self.load_data()

#     def load_data(self):
#         with open(self.qrel_path, 'r') as f: self.qrels = json.load(f)
#         with open(self.sem_path, 'r') as f: self.sem_run = json.load(f)
#         with open(self.str_path, 'r') as f: self.str_run = json.load(f)
#         with open("data/processed/queries_full.json", 'r') as f: self.queries = json.load(f)

#     def calculate_metrics(self, run_dict):
#         metrics = {"P@1": [], "MRR": [], "nDCG@10": [], "MAP": []}
#         for qid, target_docs in self.qrels.items():
#             if qid not in run_dict:
#                 for m in metrics: metrics[m].append(0)
#                 continue
            
#             # 按分数排序检索结果
#             retrieved = sorted(run_dict[qid].items(), key=lambda x: x[1], reverse=True)
#             relevant_docs = {str(k): v for k, v in target_docs.items() if v > 0}
            
#             if not relevant_docs: continue

#             # 1. P@1
#             metrics["P@1"].append(1 if retrieved[0][0] in relevant_docs else 0)

#             # 2. MRR
#             mrr = 0
#             for i, (doc_id, _) in enumerate(retrieved):
#                 if str(doc_id) in relevant_docs:
#                     mrr = 1.0 / (i + 1)
#                     break
#             metrics["MRR"].append(mrr)

#             # 3. nDCG@10
#             dcg = 0
#             for i, (doc_id, _) in enumerate(retrieved[:10]):
#                 if str(doc_id) in relevant_docs:
#                     rel = relevant_docs[str(doc_id)]
#                     dcg += rel / np.log2(i + 2)
            
#             # 计算 IDCG (理想情况下的最高得分)
#             rel_scores = sorted(relevant_docs.values(), reverse=True)
#             idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(rel_scores[:10])])
#             metrics["nDCG@10"].append(dcg / idcg if idcg > 0 else 0)

#             # 4. MAP
#             ap, hits = 0, 0
#             for i, (doc_id, _) in enumerate(retrieved):
#                 if str(doc_id) in relevant_docs:
#                     hits += 1
#                     ap += hits / (i + 1)
#             metrics["MAP"].append(ap / len(relevant_docs) if relevant_docs else 0)

#         return {k: np.mean(v) for k, v in metrics.items()}, metrics["MRR"]

#     def reciprocal_rank_fusion(self, w_sem=1.0, w_str=0.3, k_rrf=60):
#         """加权 RRF：通过降低结构流权重减少噪声"""
#         fused_run = defaultdict(dict)
#         all_qids = set(self.sem_run.keys()) | set(self.str_run.keys())
        
#         for qid in all_qids:
#             scores = defaultdict(float)
#             if qid in self.sem_run:
#                 sorted_sem = sorted(self.sem_run[qid].items(), key=lambda x: x[1], reverse=True)
#                 for rank, (doc_id, _) in enumerate(sorted_sem):
#                     scores[doc_id] += w_sem / (k_rrf + rank + 1)
#             if qid in self.str_run:
#                 sorted_str = sorted(self.str_run[qid].items(), key=lambda x: x[1], reverse=True)
#                 for rank, (doc_id, _) in enumerate(sorted_str):
#                     scores[doc_id] += w_str / (k_rrf + rank + 1)
#             fused_run[qid] = dict(scores)
#         return fused_run

#     # def run_ablation(self):
#     #     print("\n>>> 执行加权消融实验 (Weighted Ablation Study)...")
#     #     results = []
        
#     #     # S1: Semantic
#     #     m_s1, mrr_s1 = self.calculate_metrics(self.sem_run)
#     #     results.append({"Setting": "S1: Semantic only", **m_s1})
        
#     #     # S2: Structural
#     #     m_s2, _ = self.calculate_metrics(self.str_run)
#     #     results.append({"Setting": "S2: Structural only", **m_s2})
        
#     #     # S4: LS-MIR (Weighted Fusion)
#     #     fused = self.reciprocal_rank_fusion(w_sem=1.0, w_str=0.9)
#     #     m_s4, mrr_s4 = self.calculate_metrics(fused)
#     #     results.append({"Setting": "S4: LS-MIR (Proposed)", **m_s4})
        
#     #     print(tabulate(pd.DataFrame(results), headers='keys', tablefmt='pipe', floatfmt=".4f"))
        
#     #     t_stat, p_val = stats.ttest_rel(mrr_s1, mrr_s4)
#     #     print(f"\n[Statistical Significance] p-value: {p_val:.6e}")

#     # import re # 确保你在文件最上方添加了这一行
#     def run_dynamic_optimization(self):
#         print("\n>>> 正在开启动态权重搜索 (Dynamic Weight Optimization)...")
        
#         # 1. 计算基准 (S1: Semantic Only) 的 MRR 序列用于显著性校验
#         m_s1, mrr_s1_list = self.calculate_metrics(self.sem_run)
        
#         search_results = []
#         best_mrr = -1
#         best_w = 0
        
#         # 2. 遍历权重空间
#         weights = np.arange(0.1, 1.1, 0.1) # 0.1, 0.2, ..., 1.0
#         for w in weights:
#             # 执行混合检索
#             fused = self.reciprocal_rank_fusion(w_sem=1.0, w_str=w)
#             # 计算各项指标
#             metrics, mrr_list = self.calculate_metrics(fused)
            
#             # 计算显著性 (对比 S1)
#             t_stat, p_val = stats.ttest_rel(mrr_s1_list, mrr_list)
            
#             res = {
#                 "w_str": w,
#                 "P@1": metrics["P@1"],
#                 "MRR": metrics["MRR"],
#                 "p-value": p_val,
#                 "Significant": "YES" if p_val < 0.05 else "NO"
#             }
#             search_results.append(res)
            
#             # 记录最优 MRR
#             if metrics["MRR"] > best_mrr:
#                 best_mrr = metrics["MRR"]
#                 best_w = w

#         # 3. 输出搜索结果表格
#         df_res = pd.DataFrame(search_results)
#         print(tabulate(df_res, headers='keys', tablefmt='pipe', floatfmt=".4f"))
        
#         print(f"\n✅ 搜索完成！在 w_str = {best_w:.1f} 时取得最优 MRR: {best_mrr:.4f}")
#         return best_w

#     def run_dynamic_optimization(self):
#         print("\n>>> 正在开启动态权重搜索 (Dynamic Weight Optimization)...")
        
#         # 1. 计算基准 (S1: Semantic Only) 的 MRR 序列用于显著性校验
#         m_s1, mrr_s1_list = self.calculate_metrics(self.sem_run)
        
#         search_results = []
#         best_mrr = -1
#         best_w = 0
        
#         # 2. 遍历权重空间
#         weights = np.arange(0.1, 1.1, 0.1) # 0.1, 0.2, ..., 1.0
#         for w in weights:
#             # 执行混合检索
#             fused = self.reciprocal_rank_fusion(w_sem=1.0, w_str=w)
#             # 计算各项指标
#             metrics, mrr_list = self.calculate_metrics(fused)
            
#             # 计算显著性 (对比 S1)
#             t_stat, p_val = stats.ttest_rel(mrr_s1_list, mrr_list)
            
#             res = {
#                 "w_str": w,
#                 "P@1": metrics["P@1"],
#                 "MRR": metrics["MRR"],
#                 "p-value": p_val,
#                 "Significant": "YES" if p_val < 0.05 else "NO"
#             }
#             search_results.append(res)
            
#             # 记录最优 MRR
#             if metrics["MRR"] > best_mrr:
#                 best_mrr = metrics["MRR"]
#                 best_w = w

#         # 3. 输出搜索结果表格
#         df_res = pd.DataFrame(search_results)
#         print(tabulate(df_res, headers='keys', tablefmt='pipe', floatfmt=".4f"))
        
#         print(f"\n✅ 搜索完成！在 w_str = {best_w:.1f} 时取得最优 MRR: {best_mrr:.4f}")
#         return best_w

#     def run_complexity_analysis(self):
#         print("\n>>> 执行复杂度深度分析 (Token-based)...")
#         # 保持与消融实验一致的权重
#         fused = self.reciprocal_rank_fusion(w_sem=1.0, w_str=0.9)
#         complexity_res = []
        
#         # 定义复杂度区间 (Token 数量)
#         categories = {
#             "Simple (<20)": (0, 20), 
#             "Medium (20-50)": (20, 50), 
#             "Complex (>50)": (50, 9999)
#         }
        
#         # 备份原始真值表
#         original_qrels = self.qrels
        
#         for name, (low, high) in categories.items():
#             cat_qids = []
#             for qid, text in self.queries.items():
#                 if qid not in original_qrels:
#                     continue
                
#                 # --- 核心改进：使用正则统计 LaTeX Token 数 ---
#                 # 统计反斜杠命令 (\int), 单词 (x), 以及特殊算子 ({}, ^, _, +, =)
#                 tokens = re.findall(r'\\[a-zA-Z]+|[\w]+|[{}()^|_=+]', str(text))
#                 token_count = len(tokens)
                
#                 if low <= token_count < high:
#                     cat_qids.append(qid)
            
#             if not cat_qids:
#                 continue
                
#             # 提取当前类别的真值和检索结果
#             self.qrels = {qid: original_qrels[qid] for qid in cat_qids}
#             cat_run = {qid: fused[qid] for qid in cat_qids if qid in fused}
            
#             # 计算该类别的指标
#             m, _ = self.calculate_metrics(cat_run)
            
#             complexity_res.append({
#                 "Category": name, 
#                 "Count": len(cat_qids), 
#                 "MRR": m["MRR"],
#                 "P@1": m["P@1"]
#             })
        
#         # 还原真值表
#         self.qrels = original_qrels
        
#         # 输出表格
#         print(tabulate(pd.DataFrame(complexity_res), headers='keys', tablefmt='pipe', floatfmt=".4f"))

# if __name__ == "__main__":
#     evaluator = Evaluator(
#         qrel_path="data/qrel_76_expert.json",
#         sem_path="results/raw_sem_scores.json",
#         str_path="results/raw_str_scores.json"
#     )
#     evaluator.run_ablation()
#     evaluator.run_complexity_analysis()


import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from tabulate import tabulate
import re
import time # 确保在文件顶部导入了 time

class Evaluator:
    def __init__(self, qrel_path, sem_path, str_path, query_path):
        self.qrel_path = qrel_path
        self.sem_path = sem_path
        self.str_path = str_path
        self.query_path = query_path
        self.load_data()

    def load_data(self):
        print(f"📂 正在加载数据源...")
        with open(self.qrel_path, 'r') as f: self.qrels = json.load(f)
        with open(self.sem_path, 'r') as f: self.sem_run = json.load(f)
        with open(self.str_path, 'r') as f: self.str_run = json.load(f)
        with open(self.query_path, 'r') as f: self.queries = json.load(f)
        print(f"✅ 数据加载完成。有效查询数: {len(self.qrels)}")


    def run_latency_audit(self, best_w):
        """测量融合算法的工程效率 (针对 76 个查询)"""
        print("\n>>> 执行检索效率审计 (Latency Audit)...")
        
        start_time = time.time()
        # 模拟执行一次完整的融合过程
        for _ in range(10): # 运行 10 次取平均以消除系统扰动
            _ = self.reciprocal_rank_fusion(w_sem=1.0, w_str=best_w)
        
        end_time = time.time()
        
        # 计算单次融合的总耗时 (针对 76 个查询)
        total_fusion_avg = ((end_time - start_time) / 10) * 1000 # 毫秒
        # 计算单个查询的平均融合耗时
        per_query_fusion = total_fusion_avg / len(self.qrels)
        
        print(f"| 统计项 | 数值 |")
        print(f"| :--- | :--- |")
        print(f"| 评估查询总数 | {len(self.qrels)} |")
        print(f"| 单个查询平均融合耗时 (RRF) | {per_query_fusion:.2f} ms |")
        print(f"\n💡 提示：RRF 阶段耗时极低，系统总延迟的主要来源是索引检索阶段。")

    def calculate_metrics(self, run_dict):
        """计算核心 IR 指标"""
        metrics = {"P@1": [], "MRR": [], "nDCG@10": [], "MAP": []}
        
        for qid, target_docs in self.qrels.items():
            if qid not in run_dict or not run_dict[qid]:
                for m in metrics: metrics[m].append(0)
                continue
            
            # 按分数从高到低排序结果
            retrieved = sorted(run_dict[qid].items(), key=lambda x: x[1], reverse=True)
            relevant_docs = {str(k): v for k, v in target_docs.items() if v > 0}
            
            if not relevant_docs: continue

            # 1. P@1
            metrics["P@1"].append(1 if str(retrieved[0][0]) in relevant_docs else 0)

            # 2. MRR
            mrr = 0
            for i, (doc_id, _) in enumerate(retrieved):
                if str(doc_id) in relevant_docs:
                    mrr = 1.0 / (i + 1)
                    break
            metrics["MRR"].append(mrr)

            # 3. nDCG@10
            dcg = 0
            for i, (doc_id, _) in enumerate(retrieved[:10]):
                if str(doc_id) in relevant_docs:
                    rel = relevant_docs[str(doc_id)]
                    dcg += rel / np.log2(i + 2)
            
            rel_scores = sorted(relevant_docs.values(), reverse=True)
            idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(rel_scores[:10])])
            metrics["nDCG@10"].append(dcg / idcg if idcg > 0 else 0)

            # 4. MAP
            ap, hits = 0, 0
            for i, (doc_id, _) in enumerate(retrieved):
                if str(doc_id) in relevant_docs:
                    hits += 1
                    ap += hits / (i + 1)
            metrics["MAP"].append(ap / len(relevant_docs) if relevant_docs else 0)

        return {k: np.mean(v) for k, v in metrics.items()}, metrics["MRR"]

    def reciprocal_rank_fusion(self, w_sem=1.0, w_str=0.3, k_rrf=60):
        """加权 RRF 融合逻辑"""
        fused_run = defaultdict(dict)
        all_qids = set(self.sem_run.keys()) | set(self.str_run.keys())
        
        for qid in all_qids:
            scores = defaultdict(float)
            # 处理语义流
            if qid in self.sem_run:
                sorted_sem = sorted(self.sem_run[qid].items(), key=lambda x: x[1], reverse=True)
                for rank, (doc_id, _) in enumerate(sorted_sem[:1000]):
                    scores[doc_id] += w_sem / (k_rrf + rank + 1)
            # 处理结构流
            if qid in self.str_run:
                sorted_str = sorted(self.str_run[qid].items(), key=lambda x: x[1], reverse=True)
                for rank, (doc_id, _) in enumerate(sorted_str[:1000]):
                    scores[doc_id] += w_str / (k_rrf + rank + 1)
            fused_run[qid] = dict(scores)
        return fused_run

    def run_dynamic_optimization(self):
        """动态超参数搜索：寻找性能与显著性的平衡点"""
        print("\n>>> 正在开启动态权重搜索 (Grid Search for w_str)...")
        m_s1, mrr_s1_list = self.calculate_metrics(self.sem_run)
        
        search_results = []
        best_mrr = -1
        optimal_w = 0
        
        weights = np.arange(0.1, 1.1, 0.1)
        for w in weights:
            fused = self.reciprocal_rank_fusion(w_sem=1.0, w_str=w)
            metrics, mrr_list = self.calculate_metrics(fused)
            _, p_val = stats.ttest_rel(mrr_s1_list, mrr_list)
            
            res = {
                "w_str": round(w, 1),
                "P@1": metrics["P@1"],
                "MRR": metrics["MRR"],
                "nDCG@10": metrics["nDCG@10"],
                "p-value": p_val,
                "Sig (<0.05)": "✅" if p_val < 0.05 else "❌"
            }
            search_results.append(res)
            
            if metrics["MRR"] > best_mrr:
                best_mrr = metrics["MRR"]
                optimal_w = w

        print(tabulate(pd.DataFrame(search_results), headers='keys', tablefmt='pipe', floatfmt=".4f"))
        print(f"\n💡 建议最优权重: w_str = {optimal_w:.1f} (MRR: {best_mrr:.4f})")
        return optimal_w

    def run_complexity_analysis(self, best_w):
        """基于 Token 长度的复杂度深度分析"""
        print(f"\n>>> 执行复杂度深度分析 (Token-based, w_str={best_w:.1f})...")
        fused = self.reciprocal_rank_fusion(w_sem=1.0, w_str=best_w)
        
        categories = {
            "Simple (<20)": (0, 20), 
            "Medium (20-50)": (20, 50), 
            "Complex (>50)": (50, 999)
        }
        
        original_qrels = self.qrels
        complexity_res = []
        
        for name, (low, high) in categories.items():
            cat_qids = []
            for qid, query_obj in self.queries.items():
                # 兼容不同 JSON 格式
                text = query_obj['latex'] if isinstance(query_obj, dict) else str(query_obj)
                if qid not in original_qrels: continue
                
                # Token 统计正则
                tokens = re.findall(r'\\[a-zA-Z]+|[\w]+|[{}()^|_=+]', text)
                if low <= len(tokens) < high:
                    cat_qids.append(qid)
            
            if not cat_qids: continue
            
            self.qrels = {qid: original_qrels[qid] for qid in cat_qids}
            cat_run = {qid: fused[qid] for qid in cat_qids}
            m, _ = self.calculate_metrics(cat_run)
            
            complexity_res.append({
                "Category": name, "Count": len(cat_qids), 
                "MRR": m["MRR"], "P@1": m["P@1"]
            })
        
        self.qrels = original_qrels # 还原
        print(tabulate(pd.DataFrame(complexity_res), headers='keys', tablefmt='pipe', floatfmt=".4f"))

if __name__ == "__main__":
    # 初始化评估器
    evaluator = Evaluator(
        qrel_path="data/qrel_76_expert.json",
        sem_path="results/raw_sem_scores.json",
        str_path="results/raw_str_scores.json",
        query_path="data/processed/queries_full.json"
    )
    
    # 1. 运行动态搜索，找到最佳权重
    best_w = evaluator.run_dynamic_optimization()
    
    # 2. 基于最佳权重运行复杂度分析
    evaluator.run_complexity_analysis(best_w)
    # 3. 新增：运行效率审计
    evaluator.run_latency_audit(best_w)

