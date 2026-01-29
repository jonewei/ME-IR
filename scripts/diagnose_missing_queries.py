import json
import os

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERY_PATH = os.path.join(PROJECT_ROOT, "data/processed/queries_full.json")
RELEVANCE_PATH = os.path.join(PROJECT_ROOT, "data/processed/relevance_labels.json")

def diagnose():
    with open(QUERY_PATH, 'r') as f:
        queries = json.load(f)
    with open(RELEVANCE_PATH, 'r') as f:
        relevance = json.load(f)

    all_qids = set(queries.keys())
    relevant_qids = set(relevance.keys())

    # 1. 找出完全没有标注的 Query
    missing_qids = all_qids - relevant_qids
    
    print(f"📋 总查询数: {len(all_qids)}")
    print(f"✅ 有标注的查询数: {len(relevant_qids)}")
    print(f"❌ 缺失标注（被过滤）的查询数: {len(missing_qids)}")
    print("-" * 50)

    # 2. 输出具体的缺失清单
    print(f"{'QID':<15} | {'LaTeX Content (Snippet)'}")
    print("-" * 50)
    
    for qid in sorted(list(missing_qids)):
        latex = queries[qid]
        # 只显示前 50 个字符方便人工检查
        display_latex = (latex[:50] + '...') if len(latex) > 50 else latex
        print(f"{qid:<15} | {display_latex}")

    # 3. 检查解析失败的情况 (针对那 76 条之内的)
    # 我们看这 76 条里是否有查询在子结构索引中提取不到任何路径
    from retrieval.path_inverted_index import PathInvertedIndex
    sub_index = PathInvertedIndex()
    
    parsing_fails = []
    for qid in relevant_qids:
        if qid in queries:
            paths = sub_index._extract_paths(queries[qid])
            if not paths:
                parsing_fails.append(qid)
                
    if parsing_fails:
        print("\n⚠️ 注意：以下有标注的查询虽然没被过滤，但 LaTeX 解析失败（提取不到路径）：")
        for qid in parsing_fails:
            print(f"- {qid}: {queries[qid]}")
    else:
        print("\n⭐ 所有的 76 条有标注查询均成功解析并提取了结构路径。")

if __name__ == "__main__":
    diagnose()