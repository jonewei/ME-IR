import json
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# 路径配置
FORMULAS_JSON = "data/processed/formulas.json"
QUERIES_JSON = "data/processed/queries_full.json"
LATEX_DIR = "data/arqmath3/latex_representation_v3"

def get_visual_id_frequencies():
    """统计每个 Visual ID 在 2826 万原始实例中出现的频率"""
    print("📊 正在统计原始语料中的 Visual ID 频率分布...")
    freq_map = Counter()
    tsv_files = sorted(list(Path(LATEX_DIR).glob("*.tsv")))
    
    for f in tqdm(tsv_files, desc="Scanning for frequencies"):
        with open(f, 'r', encoding='utf-8') as fin:
            reader = csv.reader(fin, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) > 6:
                    freq_map[row[6].strip()] += 1
    return freq_map

def analyze_diversity():
    # 1. 准备数据
    freq_map = get_visual_id_frequencies()
    
    # 加载已有的评测逻辑
    from evaluation.final_hybrid_evaluator import HybridEvaluator
    evaluator = HybridEvaluator()
    
    TOP_K = 100
    results_report = []

    print(f"\n🚀 开始对 Top-{TOP_K} 结果进行多样性建模...")
    
    # 随机取 20 条查询进行深度分析
    sample_queries = list(evaluator.queries.items())[:20]
    
    for qid, q_latex in tqdm(sample_queries, desc="Analyzing Queries"):
        # 获取去重后的真实搜索结果 (Visual IDs)
        dedup_results = evaluator.search_single(q_latex)[:TOP_K]
        
        # 计算“冗余压力”：如果没去重，这些结果会占据多少空间？
        # 例如：排名前 10 的公式如果每个都重复了 5 次，那它们会挤占前 50 个排名
        total_slots_consumed = 0
        expanded_rank_at_10 = 0
        
        for i, vid in enumerate(dedup_results):
            freq = freq_map.get(vid, 1)
            total_slots_consumed += freq
            if i == 9: # 记录前 10 名被挤压到了什么位置
                expanded_rank_at_10 = total_slots_consumed

        # 计算“有效信息增益”
        # 在 8.4M 索引中，Top-100 能给用户展示 100 种不同的数学思路
        # 在 28M 索引中，Top-100 可能只能展示 100/avg_freq 种思路
        diversity_gain = TOP_K / (total_slots_consumed / TOP_K)
        
        results_report.append({
            "qid": qid,
            "dedup_unique_count": len(dedup_results),
            "simulated_slots": total_slots_consumed,
            "rank_inflation": total_slots_consumed / TOP_K,
            "expanded_rank_at_10": expanded_rank_at_10
        })

    # 3. 输出多样性报告
    print("\n" + "="*60)
    print("📈 检索多样性与排名优化报告")
    print("="*60)
    
    avg_inflation = np.mean([r['rank_inflation'] for r in results_report])
    avg_rank_10 = np.mean([r['expanded_rank_at_10'] for r in results_report])
    
    print(f"1. 平均排名通胀率 (Rank Inflation): {avg_inflation:.2f}x")
    print(f"   [解释]: 如果不去重，搜索结果中的冗余会使列表长度膨胀 {avg_inflation:.2f} 倍。")
    print("-" * 60)
    print(f"2. 前 10 名的视觉挤压 (Top-10 Compression):")
    print(f"   [结论]: 去重后的前 10 个公式，在原始语料中平均占据了前 {avg_rank_10:.1f} 个槽位。")
    print("-" * 60)
    print(f"3. 核心贡献 (Key Contribution):")
    print(f"   去重逻辑为用户在 Top-100 窗口内多释放了 {int(TOP_K * (avg_inflation-1))} 个有效信息槽位。")
    print("="*60)

if __name__ == "__main__":
    analyze_diversity()