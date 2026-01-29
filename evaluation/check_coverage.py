import json
from pathlib import Path

def check_coverage():
    # 1. 指向你刚刚跑完的 10 个分片的映射文件
    mapping_path = Path("artifacts/vector_id_mapping_full_4090.json")
    label_path = Path("data/processed/relevance_labels.json")

    if not mapping_path.exists() or not label_path.exists():
        print("❌ 错误: 找不到映射文件或标注文件，请检查路径。")
        return

    # 2. 加载你索引库中现有的所有 ID
    with open(mapping_path, 'r') as f:
        indexed_fids = set(json.load(f))
    print(f"✅ 已加载索引库，包含 {len(indexed_fids):,} 条公式。")

    # 3. 加载标准标注 (Ground Truth)
    with open(label_path, 'r') as f:
        relevance = json.load(f)
    
    total_gt_docs = 0
    available_gt_docs = 0
    queries_with_gt = 0
    queries_with_zero_gt_in_index = 0

    for topic_id, docs in relevance.items():
        query_gt_count = len(docs)
        total_gt_docs += query_gt_count
        
        # 统计该查询的相关文档有多少在你当前的索引库里
        hits_in_index = sum(1 for doc_id in docs.keys() if str(doc_id) in indexed_fids)
        available_gt_docs += hits_in_index
        
        if query_gt_count > 0:
            queries_with_gt += 1
            if hits_in_index == 0:
                queries_with_zero_gt_in_index += 1

    print("\n📊 --- 数据覆盖率诊断报告 ---")
    print(f"1. 标注库总计相关文档数: {total_gt_docs}")
    print(f"2. 当前索引库(10分片)包含的相关文档数: {available_gt_docs}")
    print(f"3. 理论最高 Recall@1000 上限: {available_gt_docs/total_gt_docs:.2%}")
    print("-" * 40)
    print(f"4. 总共有标注的查询数: {queries_with_gt}")
    print(f"5. 在当前库中『一个答案都没有』的查询数: {queries_with_zero_gt_in_index}")
    
    if available_gt_docs == 0:
        print("\n⚠️ 警报: 你的索引库里完全没有标准答案！可能是 ID 格式不匹配（如 '123' vs 123）。")
    elif available_gt_docs < total_gt_docs * 0.05:
        print("\n💡 结论: 覆盖率太低。建议直接跑 101 个全量分片，否则 Recall 永远上不去。")
    else:
        print("\n💡 结论: 覆盖率尚可，如果 Recall 依然是 0.01%，说明 MathBERT 的语义索引效果极差，需要微调。")

if __name__ == "__main__":
    check_coverage()