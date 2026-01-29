# import json
# import pickle
# import faiss
# import numpy as np
# import torch
# from pathlib import Path
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# from retrieval.approach0_hash import DualHashGenerator, Approach0HashIndex

# # ==================== 核心配置 ====================
# MODEL_NAME = 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath'
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TOP_K = 1000

# # 路径配置
# HASH_INDEX_PATH = "artifacts/approach0_index.pkl"
# VECTOR_INDEX_PATH = "artifacts/vector_index_full_v3.faiss"
# VECTOR_MAPPING_PATH = "artifacts/vector_id_mapping_v3.json"
# FORMULAS_JSON = "data/processed/formulas.json"
# QUERIES_JSON = "data/processed/queries_full.json"
# LABELS_JSON = "data/processed/relevance_labels.json"

# class HybridEvaluator:
#     def __init__(self):
#         print("📦 正在加载双路检索系统资源...")
#         self.hash_gen = DualHashGenerator()
        
#         # 1. 加载哈希索引
#         self.h_index = Approach0HashIndex()
#         self.h_index.load(HASH_INDEX_PATH)
        
#         # 2. 加载向量索引
#         self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
#         self.v_index = faiss.read_index(VECTOR_INDEX_PATH)
#         with open(VECTOR_MAPPING_PATH, 'r') as f:
#             self.v_mapping = json.load(f)
            
#         # 3. 加载评测数据
#         with open(QUERIES_JSON, 'r') as f:
#             self.queries = json.load(f)
#         with open(LABELS_JSON, 'r') as f:
#             self.relevance = json.load(f)
        
#         print(f"✅ 资源加载完成。索引规模: {self.v_index.ntotal:,}")

#     def search_single(self, query_latex):
#         """双路召回并合并"""
#         # A. 预处理查询
#         res = self.hash_gen.clean_latex(query_latex)
        
#         # 兼容性处理：如果是元组取第一个，如果是字符串直接用
#         if isinstance(res, tuple):
#             norm_latex = res[0]
#         else:
#             norm_latex = res
        
#         # B. 第一路：哈希检索 (Stage 1)
#         h_val = self.hash_gen.generate_latex_hash(norm_latex)
#         hash_results = self.h_index.search(h_val) # 返回的是 visual_id 列表
        
#         # C. 第二路：向量检索 (Stage 2)
#         q_emb = self.model.encode(
#             [norm_latex], 
#             normalize_embeddings=True, 
#             show_progress_bar=False,
#             convert_to_numpy=True
#         ).astype('float32')
#         _, v_indices = self.v_index.search(q_emb, TOP_K)
#         vector_results = [str(self.v_mapping[idx]) for idx in v_indices[0] if idx != -1]
        
#         # D. 结果合并与去重 (哈希优先策略)
#         # 理由：哈希命中的通常是精确匹配，置信度最高
#         combined_results = []
#         seen = set()
        
#         for vid in hash_results + vector_results:
#             if vid not in seen:
#                 combined_results.append(vid)
#                 seen.add(vid)
        
#         return combined_results[:TOP_K]

#     def run_evaluation(self):
#         print(f"\n🚀 开始评测 {len(self.queries)} 条查询任务...")
        
#         recall_at_k = []
#         mrr_scores = []
        
#         # 为了更精细的分析，记录每路贡献
#         hash_only_hits = 0
#         vector_only_hits = 0
#         both_hits = 0

#         for qid, query_latex in tqdm(self.queries.items(), desc="Evaluating"):
#             gt_dict = self.relevance.get(qid, {})
#             if not gt_dict: continue
            
#             # 标准答案集合
#             gt_ids = set(str(vid) for vid in gt_dict.keys())
            
#             # 执行混合检索
#             results = self.search_single(query_latex)
            
#             # 计算指标
#             hits = gt_ids.intersection(set(results))
#             num_hits = len(hits)
            
#             # Recall@K
#             recall = num_hits / len(gt_ids) if len(gt_ids) > 0 else 0
#             recall_at_k.append(recall)
            
#             # MRR (Mean Reciprocal Rank)
#             mrr = 0
#             for rank, res_id in enumerate(results):
#                 if res_id in gt_ids:
#                     mrr = 1 / (rank + 1)
#                     break
#             mrr_scores.append(mrr)

#         # 打印最终报告
#         mean_recall = np.mean(recall_at_k) * 100
#         mean_mrr = np.mean(mrr_scores)
        
#         print("\n" + "="*60)
#         print("🏆 ARQMATH-3 混合检索评测报告")
#         print("="*60)
#         print(f"📊 基础指标:")
#         print(f"   Mean Recall@{TOP_K}: {mean_recall:.2f}%")
#         print(f"   Mean MRR@{TOP_K}:    {mean_mrr:.4f}")
#         print("-" * 60)
#         print(f"💡 调试分析:")
#         print(f"   总计评测查询数: {len(recall_at_k)}")
#         print(f"   配置模型: {MODEL_NAME}")
#         print(f"   对齐策略: Visual-ID Deduplication")
#         print("="*60)

# if __name__ == "__main__":
#     evaluator = HybridEvaluator()
#     evaluator.run_evaluation()
import json
import faiss
import numpy as np
from tqdm import tqdm
from retrieval.approach0_hash import DualHashGenerator, Approach0HashIndex

class HybridEvaluator:
    def __init__(self):
        print("📦 加载检索资源...")
        self.hash_gen = DualHashGenerator()
        self.h_index = Approach0HashIndex()
        self.h_index.load("artifacts/approach0_index.pkl")
        
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('math-similarity/Bert-MLM_arXiv-MP-class_zbMath', device="cuda")
        self.v_index = faiss.read_index("artifacts/vector_index_full_v4.faiss")
        with open("artifacts/vector_id_mapping_v4.json", 'r') as f:
            self.v_mapping = json.load(f)
            
        with open("data/processed/queries_full.json", 'r') as f:
            self.queries = json.load(f)
        with open("data/processed/relevance_labels.json", 'r') as f:
            self.relevance = json.load(f)

    def search_single(self, query_latex):
        # A. 预处理 (适配多返回值)
        res = self.hash_gen.clean_latex(query_latex)
        norm_latex = res[0] if isinstance(res, tuple) else res
        
        # B. 哈希路
        h_val = self.hash_gen.generate_latex_hash(norm_latex)
        h_res = self.h_index.search(h_val)
        
        # C. 向量路
        q_emb = self.model.encode([norm_latex], normalize_embeddings=True, convert_to_numpy=True)
        _, v_indices = self.v_index.search(q_emb.astype('float32'), 1000)
        v_res = [str(self.v_mapping[idx]) for idx in v_indices[0] if idx != -1]
        
        # D. 合并
        combined = []
        seen = set()
        for vid in h_res + v_res:
            if vid not in seen:
                combined.append(vid)
                seen.add(vid)
        return combined[:1000]

    def run(self):
        recalls, mrr_scores = [], []
        for qid, q_latex in tqdm(self.queries.items(), desc="Evaluating"):
            gt = set(str(k) for k in self.relevance.get(qid, {}).keys())
            if not gt: continue
            
            results = self.search_single(q_latex)
            hits = gt.intersection(set(results))
            recalls.append(len(hits)/len(gt))
            
            mrr = 0
            for i, r in enumerate(results):
                if r in gt:
                    mrr = 1/(i+1)
                    break
            mrr_scores.append(mrr)
            
        print(f"\n🏆 Mean Recall@1000: {np.mean(recalls)*100:.2f}%")
        print(f"🏆 Mean MRR@1000:    {np.mean(mrr_scores):.4f}")

if __name__ == "__main__":
    HybridEvaluator().run()
