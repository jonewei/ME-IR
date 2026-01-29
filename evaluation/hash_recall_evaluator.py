import json
import sqlite3
import re
from pathlib import Path
from tqdm import tqdm
from retrieval.approach0_hash import DualHashGenerator

# ==================== 配置 ====================
DB_PATH = "artifacts/formula_index.db"
LABEL_PATH = "data/processed/relevance_labels.json"
QUERY_PATH = "data/processed/queries_full.json"
FORMULAS_PATH = "data/processed/formulas.json"
TOP_K = 10000  # Stage 1通常召回更多候选

# =========================== 统一的LaTeX清洗函数 ===========================
def clean_latex(latex_str):
    """必须与其他脚本保持一致"""
    if not latex_str: 
        return ""
    latex_str = re.sub(r'\$\$?|\\\[|\\\]', '', latex_str)
    latex_str = re.sub(r'\\dfrac|\\tfrac', r'\\frac', latex_str)
    latex_str = re.sub(r'\\left|\\right', '', latex_str)
    latex_str = re.sub(r'\s+', ' ', latex_str.strip())
    return latex_str.lower()

# =========================== Stage 1 评测引擎 ===========================
class HashEvaluator:
    def __init__(self):
        print(f"📦 正在加载Stage 1评测环境...")
        
        # 加载数据库
        self.conn = sqlite3.connect(DB_PATH)
        self.hash_gen = DualHashGenerator()
        
        # 加载公式元数据（用于提取DNA）
        print(f"   - 正在加载公式元数据...")
        with open(FORMULAS_PATH, 'r') as f:
            self.formulas = json.load(f)
        
        # 加载查询
        with open(QUERY_PATH, 'r') as f:
            queries_raw = json.load(f)
        
        self.queries = {}
        for qid, qdata in queries_raw.items():
            if isinstance(qdata, dict):
                latex = qdata.get('latex_norm') or qdata.get('latex', '')
            else:
                latex = qdata
            self.queries[qid] = clean_latex(latex)
        
        # 加载标准答案
        with open(LABEL_PATH, 'r') as f:
            self.relevance = json.load(f)
        
        print(f"   ✅ 加载完成")
        print(f"      - 数据库: {DB_PATH}")
        print(f"      - 查询数: {len(self.queries)}")
        print(f"      - 公式库: {len(self.formulas):,}")

    def search_by_hash(self, query_latex, query_topic_id=None):
        """
        使用LaTeX哈希检索
        注意：这里只使用LaTeX哈希，因为查询没有MathML/DNA信息
        """
        # 生成查询的LaTeX哈希
        q_hash = self.hash_gen.generate_latex_hash(query_latex)
        
        # 从数据库检索匹配的公式ID
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT formula_id FROM formula_index WHERE h_latex = ? LIMIT ?',
            (q_hash, TOP_K)
        )
        results = [row[0] for row in cursor.fetchall()]
        
        return results

    def run_evaluation(self):
        """执行Stage 1评测"""
        
        recall_scores = []
        query_details = []
        
        print(f"\n🔍 开始Stage 1 (哈希检索) 评测...")
        print(f"   召回上限: Top-{TOP_K}")
        
        for topic_id, query_latex in tqdm(self.queries.items(), desc="Processing"):
            # 获取标准答案
            gt_docs = set(self.relevance.get(topic_id, {}).keys())
            if not gt_docs:
                continue
            
            # 执行哈希检索
            retrieved_ids = self.search_by_hash(query_latex, topic_id)
            retrieved_set = set(str(x) for x in retrieved_ids)
            gt_set = set(str(x) for x in gt_docs)
            
            # 计算Recall
            hits = len(gt_set.intersection(retrieved_set))
            recall = hits / len(gt_set) if len(gt_set) > 0 else 0
            
            recall_scores.append(recall)
            
            query_details.append({
                'topic_id': topic_id,
                'query': query_latex[:100],
                'gt_count': len(gt_set),
                'retrieved_count': len(retrieved_set),
                'hits': hits,
                'recall': recall
            })
        
        # 输出结果
        avg_recall = sum(recall_scores) / len(recall_scores) * 100 if recall_scores else 0
        
        print("\n" + "="*60)
        print(f"🏆 Stage 1 (结构哈希) 评测结果")
        print("="*60)
        print(f"检索方法: LaTeX Hash (MD5)")
        print(f"数据库规模: {len(self.formulas):,} 条公式")
        print(f"查询数量: {len(recall_scores)}")
        print(f"-" * 60)
        print(f"Mean Recall@{TOP_K}: {avg_recall:.2f}%")
        print("="*60)
        
        # 召回率分布
        print(f"\n📈 召回率分布:")
        bins = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
        bin_names = ["0%", "0-1%", "1-10%", "10-30%", "30-50%", "50-70%", "70-100%"]
        for i in range(len(bins)-1):
            count = sum(1 for r in recall_scores if bins[i] <= r < bins[i+1])
            pct = count / len(recall_scores) * 100
            print(f"   {bin_names[i+1]}: {count:3d} queries ({pct:5.1f}%)")
        
        # 最佳和最差案例
        sorted_details = sorted(query_details, key=lambda x: x['recall'], reverse=True)
        
        print(f"\n✅ Top 3 最佳召回:")
        for d in sorted_details[:3]:
            print(f"   [{d['topic_id']}] Recall: {d['recall']*100:.1f}% ({d['hits']}/{d['gt_count']})")
            print(f"      Query: {d['query']}")
        
        print(f"\n❌ Top 3 最差召回:")
        for d in sorted_details[-3:]:
            print(f"   [{d['topic_id']}] Recall: {d['recall']*100:.1f}% ({d['hits']}/{d['gt_count']})")
            print(f"      Query: {d['query']}")
        
        # 保存结果
        results_path = Path("evaluation_results")
        results_path.mkdir(exist_ok=True)
        
        with open(results_path / "hash_recall_details.json", 'w') as f:
            json.dump({
                'summary': {
                    'mean_recall': avg_recall,
                    'num_queries': len(recall_scores),
                    'top_k': TOP_K
                },
                'details': query_details
            }, f, indent=2)
        
        print(f"\n💾 详细结果已保存至: {results_path / 'hash_recall_details.json'}")
        
        return avg_recall

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    evaluator = HashEvaluator()
    evaluator.run_evaluation()