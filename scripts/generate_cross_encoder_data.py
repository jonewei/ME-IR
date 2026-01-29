import json
import random
from tqdm import tqdm
from evaluation.final_hybrid_evaluator import HybridEvaluator

# 配置参数
OUTPUT_FILE = "data/train_cross_encoder.jsonl"
NEGATIVES_PER_QUERY = 5  # 每个 Query 匹配 5 个难负样本
MAX_QUERIES = 500        # 用于训练的查询数（建议先用全部已标注查询）

def generate_data():
    print("🚀 启动 Day 1：难负样本挖掘流水线...")
    
    # 1. 加载资源
    evaluator = HybridEvaluator()
    with open("data/processed/formulas.json", 'r') as f:
        corpus = json.load(f)
    with open("data/processed/relevance_labels.json", 'r') as f:
        relevance = json.load(f)
    with open("data/processed/queries_full.json", 'r') as f:
        queries = json.load(f)

    train_data = []
    
    # 2. 遍历带有标注的查询
    annotated_qids = [qid for qid in queries.keys() if qid in relevance]
    
    for qid in tqdm(annotated_qids[:MAX_QUERIES], desc="Mining Hard Negatives"):
        q_latex = queries[qid]
        gt_ids = set(str(k) for k in relevance[qid].keys())
        
        if not gt_ids:
            continue

        # --- 挖掘正样本 ---
        for pos_id in gt_ids:
            if pos_id in corpus:
                train_data.append({
                    "texts": [q_latex, corpus[pos_id]],
                    "label": 1
                })

        # --- 挖掘难负样本 (核心逻辑) ---
        # 运行现有的检索系统，取 Top-50
        results = evaluator.search_single(q_latex)[:50]
        
        hard_negs = []
        for res_id in results:
            # 如果该结果不在真值库里，它就是一个“难负样本”
            if res_id not in gt_ids and res_id in corpus:
                hard_negs.append(corpus[res_id])
            
            if len(hard_negs) >= NEGATIVES_PER_QUERY:
                break
        
        for neg_latex in hard_negs:
            train_data.append({
                "texts": [q_latex, neg_latex],
                "label": 0
            })

    # 3. 保存为 JSONL 格式（方便流式读取训练）
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"\n✅ Day 1 完成！共生成 {len(train_data)} 条训练对。")
    print(f"📦 数据已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_data()