"""
级联检索评测脚本 - 修复版
修复了 torch 导入问题
"""

import json
import time
import sqlite3
import faiss
import numpy as np
import re
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from retrieval.approach0_hash import DualHashGenerator

# ==================== 配置 ====================
MODEL_NAME = 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH = "artifacts/formula_index.db"
INDEX_PATH = "artifacts/vector_index_full_v3.faiss"
MAPPING_PATH = "artifacts/vector_id_mapping_v3.json"
LABEL_PATH = "data/processed/relevance_labels.json"
QUERY_PATH = "data/processed/queries_full.json"

# Stage 1 候选集大小（可调节实验参数）
STAGE1_TOP_K = 10000
# 最终返回结果数
FINAL_TOP_K = 1000

# =========================== 统一清洗函数 ===========================
def clean_latex(latex_str):
    if not latex_str: 
        return ""
    latex_str = re.sub(r'\$\$?|\\\[|\\\]', '', latex_str)
    latex_str = re.sub(r'\\dfrac|\\tfrac', r'\\frac', latex_str)
    latex_str = re.sub(r'\\left|\\right', '', latex_str)
    latex_str = re.sub(r'\s+', ' ', latex_str.strip())
    return latex_str.lower()

# =========================== 级联检索引擎 ===========================
class CascadedRetriever:
    def __init__(self):
        print(f"📦 正在加载级联检索系统...")
        
        # Stage 1: 哈希检索
        print(f"   [Stage 1] 加载哈希数据库...")
        self.conn = sqlite3.connect(DB_PATH)
        self.hash_gen = DualHashGenerator()
        
        # Stage 2: 向量检索
        print(f"   [Stage 2] 加载向量模型与索引...")
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        self.index = faiss.read_index(INDEX_PATH)
        
        with open(MAPPING_PATH, 'r') as f:
            self.fids = json.load(f)
        
        # 创建ID到索引位置的反向映射
        self.fid_to_idx = {fid: idx for idx, fid in enumerate(self.fids)}
        
        print(f"   ✅ 级联系统加载完成")
        print(f"      - 数据库: {DB_PATH}")
        print(f"      - 向量索引: {self.index.ntotal:,} 条")

    def retrieve(self, query_latex, use_cascade=True):
        """
        执行级联检索
        """
        timing = {}
        
        if use_cascade:
            # === Stage 1: 哈希过滤 ===
            t0 = time.time()
            q_hash = self.hash_gen.generate_latex_hash(query_latex)
            
            cursor = self.conn.cursor()
            cursor.execute(
                'SELECT formula_id FROM formula_index WHERE h_latex = ? LIMIT ?',
                (q_hash, STAGE1_TOP_K)
            )
            stage1_ids = [row[0] for row in cursor.fetchall()]
            timing['stage1'] = time.time() - t0
            
            if not stage1_ids:
                use_cascade = False
            else:
                candidate_indices = [
                    self.fid_to_idx[str(fid)] 
                    for fid in stage1_ids 
                    if str(fid) in self.fid_to_idx
                ]
                
                if not candidate_indices:
                    use_cascade = False
        
        # === Stage 2: 向量重排 ===
        t0 = time.time()
        query_emb = self.model.encode(
            [query_latex], 
            normalize_embeddings=True, 
            convert_to_numpy=True
        ).astype('float32')
        
        if use_cascade and 'candidate_indices' in locals():
            # 级联模式
            candidate_vectors = np.vstack([
                self.index.reconstruct(idx) 
                for idx in candidate_indices
            ])
            
            similarities = np.dot(candidate_vectors, query_emb.T).flatten()
            top_indices = np.argsort(-similarities)[:FINAL_TOP_K]
            result_indices = [candidate_indices[i] for i in top_indices]
            result_distances = [similarities[i] for i in top_indices]
        else:
            # 全量模式
            distances, indices = self.index.search(query_emb, FINAL_TOP_K)
            result_indices = indices[0].tolist()
            result_distances = distances[0].tolist()
        
        timing['stage2'] = time.time() - t0
        
        result_ids = [self.fids[idx] for idx in result_indices if idx != -1]
        
        return result_ids, timing, result_distances

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

# =========================== 评测函数 ===========================
def run_cascaded_evaluation():
    """对比级联模式和纯向量模式"""
    
    # 加载数据
    with open(QUERY_PATH, 'r') as f:
        queries_raw = json.load(f)
    
    queries = {}
    for qid, qdata in queries_raw.items():
        if isinstance(qdata, dict):
            latex = qdata.get('latex_norm') or qdata.get('latex', '')
        else:
            latex = qdata
        queries[qid] = clean_latex(latex)
    
    with open(LABEL_PATH, 'r') as f:
        relevance = json.load(f)
    
    # 初始化检索器
    retriever = CascadedRetriever()
    
    # 存储结果
    results = {
        'cascade': {'recalls': [], 'times': []},
        'pure_vector': {'recalls': [], 'times': []}
    }
    
    print(f"\n🚀 开始级联检索评测...")
    print(f"   查询数量: {len(queries)}")
    print(f"   Stage 1 候选: {STAGE1_TOP_K}")
    print(f"   最终返回: {FINAL_TOP_K}")
    
    for topic_id, query_latex in tqdm(list(queries.items()), desc="Evaluating"):
        gt_docs = set(str(x) for x in relevance.get(topic_id, {}).keys())
        if not gt_docs:
            continue
        
        # 模式1: 级联检索
        result_ids, timing, _ = retriever.retrieve(query_latex, use_cascade=True)
        retrieved_set = set(str(x) for x in result_ids)
        hits = len(gt_docs.intersection(retrieved_set))
        recall = hits / len(gt_docs)
        results['cascade']['recalls'].append(recall)
        results['cascade']['times'].append(timing)
        
        # 模式2: 纯向量检索
        result_ids, timing, _ = retriever.retrieve(query_latex, use_cascade=False)
        retrieved_set = set(str(x) for x in result_ids)
        hits = len(gt_docs.intersection(retrieved_set))
        recall = hits / len(gt_docs)
        results['pure_vector']['recalls'].append(recall)
        results['pure_vector']['times'].append(timing)
    
    # 输出结果
    print("\n" + "="*70)
    print(f"🏆 级联检索对比评测结果")
    print("="*70)
    
    for mode_name, mode_data in results.items():
        avg_recall = np.mean(mode_data['recalls']) * 100
        avg_time = np.mean([sum(t.values()) for t in mode_data['times']]) * 1000
        
        mode_title = '级联模式 (Stage 1 + 2)' if mode_name == 'cascade' else '纯向量模式 (Stage 2 Only)'
        print(f"\n{mode_title}")
        print(f"   Mean Recall@{FINAL_TOP_K}: {avg_recall:.2f}%")
        print(f"   平均查询时间: {avg_time:.1f} ms")
        
        if mode_name == 'cascade' and mode_data['times']:
            stage1_time = np.mean([t.get('stage1', 0) for t in mode_data['times']]) * 1000
            stage2_time = np.mean([t.get('stage2', 0) for t in mode_data['times']]) * 1000
            print(f"      - Stage 1 (Hash): {stage1_time:.1f} ms")
            print(f"      - Stage 2 (Vector): {stage2_time:.1f} ms")
    
    print("="*70)
    
    # 保存结果
    results_path = Path("evaluation_results")
    results_path.mkdir(exist_ok=True)
    
    with open(results_path / "cascaded_comparison.json", 'w') as f:
        json.dump({
            mode: {
                'mean_recall': np.mean(data['recalls']) * 100,
                'std_recall': np.std(data['recalls']) * 100,
                'mean_time_ms': np.mean([sum(t.values()) for t in data['times']]) * 1000,
                'num_queries': len(data['recalls'])
            }
            for mode, data in results.items()
        }, f, indent=2)
    
    print(f"\n💾 对比结果已保存至: {results_path / 'cascaded_comparison.json'}")

if __name__ == "__main__":
    run_cascaded_evaluation()