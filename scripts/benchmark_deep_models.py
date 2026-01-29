import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

# --- 路径配置 ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERY_PATH = os.path.join(PROJECT_ROOT, "data/processed/queries_full.json")
CORPUS_PATH = os.path.join(PROJECT_ROOT, "data/processed/formulas.json")
RELEVANCE_PATH = os.path.join(PROJECT_ROOT, "data/processed/relevance_labels.json")

# --- 模型配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MATHBERT_NAME = "math-similarity/Bert-MLM_arXiv-MP-class_zbMath"

class BenchmarkEvaluator:
    def __init__(self):
        print(f"📡 正在初始化模型 (Device: {DEVICE})...")
        # 1. 加载 MiniLM
        self.minilm = SentenceTransformer(MINILM_NAME).to(DEVICE)
        # 2. 加载 Math-BERT
        self.math_tokenizer = AutoTokenizer.from_pretrained(MATHBERT_NAME)
        self.math_model = AutoModel.from_pretrained(MATHBERT_NAME).to(DEVICE)
        
        # 3. 加载数据
        with open(QUERY_PATH, 'r') as f: self.queries = json.load(f)
        with open(RELEVANCE_PATH, 'r') as f: self.relevance = json.load(f)
        with open(CORPUS_PATH, 'r') as f: self.corpus = json.load(f)
        
        # 筛选 76 条有效查询
        self.test_qids = [qid for qid in self.queries.keys() if qid in self.relevance]
        print(f"✅ 数据准备就绪，共计 {len(self.test_qids)} 条有效验证查询。")

    def get_mathbert_embedding(self, text):
        inputs = self.math_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.math_model(**inputs)
        return outputs.last_hidden_state[0][0] # CLS token

    def run_comparison(self):
        mrr_minilm = []
        mrr_mathbert = []
        
        # 为了公平对比，我们需要一组“候选池”。
        # 实际论文中，我们取 Hybrid 搜索出的 Top-100 进行重排，看语义模型能否自我修正。
        # 这里模拟每个 Query 的重排过程
        for qid in tqdm(self.test_qids, desc="Benchmarking Models"):
            q_latex = self.queries[qid]
            gt_fids = list(self.relevance[qid].keys()) # 获取所有真值 ID
            
            # 1. 模拟一个候选池 (包含真值 + 99个负样本)
            # 注意：在正式实验中，候选池应来自检索器的初步召回结果
            candidate_fids = gt_fids + list(self.corpus.keys())[:99] 
            candidate_texts = [self.corpus[str(fid)]['latex_norm'] for fid in candidate_fids]
            
            # --- MiniLM 排序 ---
            q_emb_mini = self.minilm.encode(q_latex, convert_to_tensor=True)
            c_emb_mini = self.minilm.encode(candidate_texts, convert_to_tensor=True)
            scores_mini = util.cos_sim(q_emb_mini, c_emb_mini)[0].cpu().numpy()
            
            # --- Math-BERT 排序 ---
            q_emb_math = self.get_mathbert_embedding(q_latex)
            c_embs_math = torch.stack([self.get_mathbert_embedding(t) for t in candidate_texts])
            scores_math = util.cos_sim(q_emb_math, c_embs_math)[0].cpu().numpy()
            
            # 计算排名和 MRR (真值 ID 在列表前部，即索引 0)
            def get_mrr(scores):
                # 对分数降序排列，获取原始索引
                ranked_idx = np.argsort(scores)[::-1]
                # 找到真值（索引为0）在排序后的位置
                rank = np.where(ranked_idx == 0)[0][0] + 1
                return 1.0 / rank

            mrr_minilm.append(get_mrr(scores_mini))
            mrr_mathbert.append(get_mrr(scores_math))

        print("\n" + "="*40)
        print(f"🏆 语义对标实验结果 (N={len(self.test_qids)})")
        print("-"*40)
        print(f"🔹 MiniLM (General) MRR:    {np.mean(mrr_minilm):.4f}")
        print(f"🔹 Math-BERT (Domain) MRR:  {np.mean(mrr_mathbert):.4f}")
        print(f"⭐ Hybrid (Proposed) MRR:  0.8062 (From Final Eval)")
        print("="*40)
        print("\n💡 结论：如果 Math-BERT 的 MRR 低于 Hybrid，则证明结构特征是不可或缺的。")

if __name__ == "__main__":
    evaluator = BenchmarkEvaluator()
    evaluator.run_comparison()