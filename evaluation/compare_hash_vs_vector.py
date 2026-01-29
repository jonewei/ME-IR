import json
import sqlite3
import torch
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from retrieval.approach0_hash import DualHashGenerator
from pathlib import Path

# ==================== 配置 ====================
DB_PATH = "artifacts/formula_index.db"
VECTOR_INDEX_PATH = "artifacts/vector_index_full_v3.faiss"
MAPPING_PATH = "artifacts/vector_id_mapping_v3.json"
FORMULA_JSON = "data/processed/formulas.json"
LABEL_PATH = "data/processed/relevance_labels.json"
QUERY_PATH = "data/processed/queries_full.json"
MODEL_NAME = 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath'

def clean_latex(latex):
    if not latex: return ""
    latex = re.sub(r'\$\$?|\\\[|\\\]', '', latex)
    latex = re.sub(r'\s+', ' ', latex)
    return latex.strip()

class DualPathAnalyzer:
    def __init__(self):
        print("📦 正在加载资源 (此步需消耗大量内存)...")
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self.hash_gen = DualHashGenerator()
        
        self.model = SentenceTransformer(MODEL_NAME, device="cuda")
        self.index = faiss.read_index(VECTOR_INDEX_PATH)
        with open(MAPPING_PATH, 'r') as f:
            self.fids = json.load(f)
        
        with open(LABEL_PATH, 'r') as f:
            self.relevance = json.load(f)
        with open(QUERY_PATH, 'r') as f:
            self.queries = json.load(f)
        with open(FORMULA_JSON, 'r') as f:
            self.corpus = json.load(f)

    def eval_stage1_hash(self):
        """评测 Stage 1: 结构化哈希召回率"""
        print("\n--- [Stage 1: 哈希召回评测] ---")
        recall_list = []
        for topic_id, query_latex in self.queries.items():
            gt_ids = set(self.relevance.get(topic_id, {}).keys())
            if not gt_ids: continue
            
            # 生成查询 DNA
            dna = self.hash_gen.generate(query_latex)
            
            # 从数据库中寻找 DNA 完全一致的公式
            self.cursor.execute("SELECT formula_id FROM formulas WHERE dna = ?", (dna,))
            retrieved_ids = {str(row[0]) for row in self.cursor.fetchall()}
            
            hits = len(gt_ids.intersection(retrieved_ids))
            recall = hits / len(gt_ids)
            recall_list.append(recall)
            
        print(f"✅ 哈希平均召回率 (Recall): {np.mean(recall_list)*100:.2f}%")

    def analyze_failure(self):
        """分析失败案例：为什么在库里却搜不到？"""
        print("\n--- [向量检索失败深度分析] ---")
        
        # 寻找一个标准答案在库中，但向量 Top-1000 没搜到的例子
        for topic_id, query_latex in self.queries.items():
            gt_dict = self.relevance.get(topic_id, {})
            if not gt_dict: continue
            
            # 1. 编码查询向量
            q_emb = self.model.encode([clean_latex(query_latex)], normalize_embeddings=True)[0]
            
            # 2. 执行 Top-1000 检索
            distances, indices = self.index.search(np.array([q_emb]).astype('float32'), 1000)
            retrieved_fids = {str(self.fids[idx]) for idx in indices[0] if idx != -1}
            
            # 3. 寻找一个“遗珠”：在库里（corpus）但不在检索结果里（retrieved_fids）的标准答案
            missed_gt_id = None
            for gt_id in gt_dict.keys():
                if str(gt_id) in self.corpus and str(gt_id) not in retrieved_fids:
                    missed_gt_id = str(gt_id)
                    break
            
            if missed_gt_id:
                print(f"🔍 发现典型失败案例 (Topic: {topic_id}):")
                print(f"   Query LaTeX: {query_latex}")
                
                # 获取该遗珠的 LaTeX 并计算向量距离
                gt_latex = self.corpus[missed_gt_id]['latex_norm']
                gt_emb = self.model.encode([clean_latex(gt_latex)], normalize_embeddings=True)[0]
                
                # 计算余弦相似度
                # 因为向量已归一化，点积即余弦相似度
                similarity = np.dot(q_emb, gt_emb)
                
                print(f"   Missed GT ID: {missed_gt_id}")
                print(f"   Missed GT LaTeX: {gt_latex}")
                print(f"   📉 语义相似度得分: {similarity:.4f}")
                print(f"   (注：1.0 为完美匹配，当前得分过低导致跌出 Top-1000)")
                
                # 额外检查：这两个公式的 DNA 是否一致？
                q_dna = self.hash_gen.generate(query_latex)
                gt_dna = self.hash_gen.generate(gt_latex)
                print(f"   DNA 匹配状态: {'✅ 一致' if q_dna == gt_dna else '❌ 不一致'}")
                break

if __name__ == "__main__":
    analyzer = DualPathAnalyzer()
    analyzer.eval_stage1_hash()
    analyzer.analyze_failure()