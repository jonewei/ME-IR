import json
import logging
from tqdm import tqdm
from retrieval.approach0_hash import DualHashGenerator
from retrieval.indexer import FormulaIndexer

def run_eval():
    indexer = FormulaIndexer()
    hash_gen = DualHashGenerator()
    
    # 加载处理后的数据
    try:
        with open("data/processed/queries_full.json", 'r') as f:
            queries = json.load(f)
        with open("data/processed/relevance_labels.json", 'r') as f:
            relevance = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到必要的数据文件 ({e.filename})")
        return

    total_recall = 0
    count = 0

    print(f"🚀 开始评测 {len(queries)} 条查询...")

    for qid, qdata in tqdm(queries.items(), desc="Evaluating"):
        # --- 核心修复：更灵活的 ID 匹配 ---
        # 尝试多种可能的 ID 匹配方式
        topic_id = None
        if qid in relevance:
            topic_id = qid
        elif f"B.{qid}" in relevance:
            topic_id = f"B.{qid}"
        elif qid.startswith("B.") and qid in relevance:
            topic_id = qid
        
        if not topic_id:
            continue
        
        # 获取该查询的相关文档集合
        gt = set(relevance[topic_id].keys())
        if not gt:
            continue
            
        # 生成查询哈希并从数据库召回
        h = hash_gen.get_dual_hash(qdata['latex_norm'], qdata['mathml_skel'])
        retrieved = set(indexer.retrieve(h['h_latex'], h['h_dna']))
        
        # 计算 Recall
        hits = len(gt.intersection(retrieved))
        recall_score = hits / len(gt)
        total_recall += recall_score
        count += 1

    # --- 核心修复：防止除零 ---
    if count == 0:
        print("\n❌ 评测失败: 未能匹配到任何有效的查询 ID。")
        print(f"提示: 检查查询 ID (示例: {list(queries.keys())[:2]}) "
              f"与标注 ID (示例: {list(relevance.keys())[:2]}) 是否匹配。")
    else:
        print(f"\n✅ 评测完成！")
        print(f"📊 成功匹配查询数: {count}")
        print(f"📊 平均召回率 (Mean Recall): {total_recall/count:.2%}")

if __name__ == "__main__":
    run_eval()