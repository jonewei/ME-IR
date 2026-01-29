import json
import faiss
import numpy as np
import hashlib
from tqdm import tqdm
from pathlib import Path
from retrieval.approach0_hash import DualHashGenerator, Approach0HashIndex

class AblationTester:
    def __init__(self):
        print("📦 正在加载消融实验所需资源...")
        self.hash_gen = DualHashGenerator()
        self.h_index = Approach0HashIndex()
        self.h_index.load("artifacts/approach0_index.pkl")
        
        # 仅在需要向量路时加载，节省显存
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('math-similarity/Bert-MLM_arXiv-MP-class_zbMath', device="cuda")
        self.v_index = faiss.read_index("artifacts/vector_index_full_v4.faiss")
        with open("artifacts/vector_id_mapping_v4.json", 'r') as f:
            self.v_mapping = json.load(f)
            
        with open("data/processed/queries_full.json", 'r') as f:
            self.queries = json.load(f) # 注意：这里存的是经过规范化的，我们需要原始查询
        
        # 重新读取原始查询 TSV，以测试 V1 (未规范化)
        self.raw_queries = self._load_raw_queries()
        with open("data/processed/relevance_labels.json", 'r') as f:
            self.relevance = json.load(f)

    def _load_raw_queries(self):
        import csv
        raw = {}
        with open("data/arqmath3/queries_arqmath3_task2.tsv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2: raw[row[0].strip()] = row[1].strip()
        return raw

    def run_search(self, query_latex, use_norm=True, use_hash=True, use_vector=True):
        results = []
        seen = set()

        # 1. 规范化处理
        if use_norm:
            norm_latex, _ = self.hash_gen.clean_latex(query_latex)
        else:
            # V1: 仅去除两端的 $ 和空格，不进行深度清洗
            norm_latex = query_latex.replace('$', '').strip()

        # 2. 哈希路
        if use_hash:
            h_val = hashlib.md5(norm_latex.encode('utf-8')).hexdigest()
            for vid in self.h_index.search(h_val):
                if vid not in seen:
                    results.append(vid)
                    seen.add(vid)

        # 3. 向量路
        if use_vector:
            q_emb = self.model.encode([norm_latex], normalize_embeddings=True, convert_to_numpy=True)
            _, v_indices = self.v_index.search(q_emb.astype('float32'), 1000)
            for idx in v_indices[0]:
                if idx != -1:
                    vid = str(self.v_mapping[idx])
                    if vid not in seen:
                        results.append(vid)
                        seen.add(vid)
        
        return results[:1000]

    def evaluate_variant(self, name, use_norm, use_hash, use_vector):
        print(f"\n🧪 正在测试变体 {name}...")
        recalls, mrr_scores = [], []
        
        for qid, raw_latex in tqdm(self.raw_queries.items(), desc=f"{name}"):
            gt = set(str(k) for k in self.relevance.get(qid, {}).keys())
            if not gt: continue
            
            results = self.run_search(raw_latex, use_norm, use_hash, use_vector)
            
            # 计算 Recall
            hits = gt.intersection(set(results))
            recalls.append(len(hits)/len(gt))
            
            # 计算 MRR
            mrr = 0
            for i, r in enumerate(results):
                if r in gt:
                    mrr = 1/(i+1)
                    break
            mrr_scores.append(mrr)
            
        return np.mean(recalls), np.mean(mrr_scores)

    def start_ablation(self):
        variants = [
            ("V1 (Baseline: Raw Hash)", False, True, False),
            ("V2 (Normalized Hash Only)", True, True, False),
            ("V3 (Semantic Vector Only)", True, False, True),
            ("V4 (Proposed Hybrid)", True, True, True),
        ]
        
        summary = []
        for name, norm, h_path, v_path in variants:
            r, m = self.evaluate_variant(name, norm, h_path, v_path)
            summary.append({"Variant": name, "Recall@1000": r, "MRR": m})
        
        print("\n" + "="*60)
        print("📊 消融实验最终实测结果")
        print("="*60)
        print(f"{'Variant':<30} | {'Recall@1000':<12} | {'MRR':<8}")
        print("-" * 60)
        for row in summary:
            print(f"{row['Variant']:<30} | {row['Recall@1000']:>11.2%} | {row['MRR']:>8.4f}")
        print("="*60)

if __name__ == "__main__":
    tester = AblationTester()
    tester.start_ablation()