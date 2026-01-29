import json
import torch
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# ==================== 配置参数 (必须与构建脚本一致) ====================
MODEL_NAME = 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_PATH = "artifacts/vector_index_full_v3.faiss"
MAPPING_PATH = "artifacts/vector_id_mapping_v3.json"
LABEL_PATH = "data/processed/relevance_labels.json"
QUERY_PATH = "data/processed/queries_full.json"
TOP_K = 1000

# =========================== 🔧 统一的LaTeX清洗函数 ===========================
def clean_latex(latex_str):
    """
    ⚠️  必须与prepare_final_arqmath.py和build_full_4090_v3.py完全一致！
    """
    if not latex_str: 
        return ""
    
    # 移除数学模式标记
    latex_str = re.sub(r'\$\$?|\\\[|\\\]', '', latex_str)
    
    # 标准化命令
    latex_str = re.sub(r'\\dfrac|\\tfrac', r'\\frac', latex_str)
    latex_str = re.sub(r'\\left|\\right', '', latex_str)
    
    # 压缩空格
    latex_str = re.sub(r'\s+', ' ', latex_str.strip())
    
    # 小写化
    latex_str = latex_str.lower()
    
    return latex_str

# =========================== 评测引擎 ===========================
class MathEvaluator:
    def __init__(self):
        print(f"📦 正在加载评测环境...")
        
        # 加载模型
        print(f"   - 模型: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        
        # 加载索引
        print(f"   - 索引: {INDEX_PATH}")
        self.index = faiss.read_index(INDEX_PATH)
        
        # 加载ID映射
        with open(MAPPING_PATH, 'r') as f:
            self.fids = json.load(f)
        
        # 加载标准答案
        with open(LABEL_PATH, 'r') as f:
            self.relevance = json.load(f)
        
        # 加载查询
        with open(QUERY_PATH, 'r') as f:
            queries_raw = json.load(f)
        
        # 🔧 关键修复：正确解析queries数据结构
        self.queries = {}
        for qid, qdata in queries_raw.items():
            # 检查数据结构
            if isinstance(qdata, dict):
                # 如果是字典，提取latex或latex_norm
                latex = qdata.get('latex_norm') or qdata.get('latex', '')
            elif isinstance(qdata, str):
                # 如果直接是字符串
                latex = qdata
            else:
                print(f"   ⚠️  警告: 查询 {qid} 的数据格式异常")
                continue
            
            # 🔧 关键修复：对查询也进行同样的清洗
            self.queries[qid] = clean_latex(latex)
        
        print(f"   ✅ 加载完成")
        print(f"      - 索引向量数: {self.index.ntotal:,}")
        print(f"      - 查询数: {len(self.queries)}")
        print(f"      - 标准答案数: {len(self.relevance)}")

    def run_evaluation(self, save_results=True):
        """执行完整评测"""
        
        # 1. 准备查询数据
        topic_ids = []
        query_texts = []
        
        for tid, latex in self.queries.items():
            topic_ids.append(tid)
            query_texts.append(latex)
        
        print(f"\n🔍 正在编码 {len(topic_ids)} 条查询公式...")
        
        # 数据质量检查
        print(f"\n📊 查询数据质量检查:")
        for i in range(min(3, len(query_texts))):
            print(f"   [{topic_ids[i]}]: {query_texts[i][:60]}...")
        
        # 🔧 关键修复：确保normalize_embeddings=True
        query_embs = self.model.encode(
            query_texts, 
            batch_size=32, 
            normalize_embeddings=True,  # 必须与索引端一致
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype('float32')

        print(f"\n⚡ 正在检索 Top-{TOP_K}...")
        distances, indices = self.index.search(query_embs, TOP_K)

        # 2. 计算Recall指标
        recall_scores = []
        precision_scores = []
        query_details = []
        
        print(f"\n📊 正在计算评测指标...")
        for i, topic_id in enumerate(tqdm(topic_ids, desc="Processing")):
            # 获取标准答案集合
            gt_docs = set(self.relevance.get(topic_id, {}).keys())
            if not gt_docs:
                continue
            
            # 获取检索结果
            retrieved_indices = indices[i]
            retrieved_fids = [str(self.fids[idx]) for idx in retrieved_indices if idx != -1]
            
            # 统一ID格式（处理可能的int/str不一致）
            retrieved_set = set(retrieved_fids)
            gt_set = set(str(x) for x in gt_docs)
            
            # 计算指标
            hits = len(gt_set.intersection(retrieved_set))
            recall = hits / len(gt_set) if len(gt_set) > 0 else 0
            precision = hits / len(retrieved_set) if len(retrieved_set) > 0 else 0
            
            recall_scores.append(recall)
            precision_scores.append(precision)
            
            # 保存详细信息用于错误分析
            query_details.append({
                'topic_id': topic_id,
                'query': query_texts[i][:100],
                'gt_count': len(gt_set),
                'retrieved_count': len(retrieved_set),
                'hits': hits,
                'recall': recall,
                'precision': precision,
                'top5_distances': distances[i][:5].tolist(),
                'top5_ids': retrieved_fids[:5]
            })

        # 3. 输出结果
        avg_recall = np.mean(recall_scores) * 100
        avg_precision = np.mean(precision_scores) * 100
        
        print("\n" + "="*60)
        print(f"🏆 向量检索评测结果 (Stage 2 Only)")
        print("="*60)
        print(f"模型: {MODEL_NAME}")
        print(f"索引规模: {self.index.ntotal:,} 条公式")
        print(f"查询数量: {len(recall_scores)}")
        print(f"-" * 60)
        print(f"Mean Recall@{TOP_K}:    {avg_recall:.2f}%")
        print(f"Mean Precision@{TOP_K}: {avg_precision:.2f}%")
        print("="*60)
        
        # 4. 错误分析
        print(f"\n📈 召回率分布:")
        bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        bin_names = ["0%", "0-1%", "1-5%", "5-10%", "10-20%", "20-50%", "50-100%"]
        for i in range(len(bins)-1):
            count = sum(1 for r in recall_scores if bins[i] <= r < bins[i+1])
            pct = count / len(recall_scores) * 100
            print(f"   {bin_names[i+1]}: {count:3d} queries ({pct:5.1f}%)")
        
        # 5. 展示最好和最差的案例
        sorted_details = sorted(query_details, key=lambda x: x['recall'], reverse=True)
        
        print(f"\n✅ Top 3 最佳召回案例:")
        for detail in sorted_details[:3]:
            print(f"\n   Topic: {detail['topic_id']}")
            print(f"   Query: {detail['query']}")
            print(f"   Recall: {detail['recall']*100:.1f}% ({detail['hits']}/{detail['gt_count']})")
            print(f"   Top-1 距离: {detail['top5_distances'][0]:.4f}")
        
        print(f"\n❌ Top 3 最差召回案例:")
        for detail in sorted_details[-3:]:
            print(f"\n   Topic: {detail['topic_id']}")
            print(f"   Query: {detail['query']}")
            print(f"   Recall: {detail['recall']*100:.1f}% ({detail['hits']}/{detail['gt_count']})")
            print(f"   Top-1 距离: {detail['top5_distances'][0]:.4f}")
            print(f"   Top-5 IDs: {detail['top5_ids']}")
        
        # 6. 保存详细结果
        if save_results:
            results_path = Path("evaluation_results")
            results_path.mkdir(exist_ok=True)
            
            with open(results_path / "vector_recall_details.json", 'w') as f:
                json.dump({
                    'summary': {
                        'mean_recall': avg_recall,
                        'mean_precision': avg_precision,
                        'num_queries': len(recall_scores),
                        'index_size': self.index.ntotal
                    },
                    'details': query_details
                }, f, indent=2)
            
            print(f"\n💾 详细结果已保存至: {results_path / 'vector_recall_details.json'}")
        
        return avg_recall

if __name__ == "__main__":
    evaluator = MathEvaluator()
    evaluator.run_evaluation()