import json
import os

# 配置路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(PROJECT_ROOT, "data/processed/formulas.json")
QUERY_PATH = os.path.join(PROJECT_ROOT, "data/processed/queries_full.json")
RELEVANCE_PATH = os.path.join(PROJECT_ROOT, "data/processed/relevance_labels.json")

def audit():
    print("🔍 正在开启学术审计模式，核实数据过滤逻辑...\n")

    # 1. 加载所有原始数据
    with open(QUERY_PATH, 'r') as f:
        all_queries = json.load(f)
    with open(RELEVANCE_PATH, 'r') as f:
        relevance = json.load(f)
    
    # 为了检查“库外真值”，我们需要加载公式库（这可能比较慢，请耐心等待）
    print("📂 正在加载 8.41M 公式库索引（用于核实真值是否存在）...")
    with open(CORPUS_PATH, 'r') as f:
        corpus_ids = set(json.load(f).keys())

    stats = {
        "missing_label": [],  # 缺失官方标注
        "out_of_corpus": [],  # 真值不在 8.41M 库里
        "parsing_error": [],  # 解析失败（本脚本模拟解析逻辑）
        "valid": []           # 最终通过的 76 条
    }

    # 2. 模拟检索系统的过滤逻辑
    from retrieval.path_inverted_index import PathInvertedIndex
    sub_index = PathInvertedIndex()

    for qid, latex in all_queries.items():
        # A. 检查是否有标注
        if qid not in relevance:
            stats["missing_label"].append(qid)
            continue
        
        # B. 检查标注的真值公式 ID 是否在我们的 8.41M 库里
        gt_fids = list(relevance[qid].keys())
        # 只要有一个真值在库里，我们就认为这个 Query 是“库内有解”的
        exists_in_corpus = any(str(fid) in corpus_ids for fid in gt_fids)
        
        if not exists_in_corpus:
            stats["out_of_corpus"].append(qid)
            continue

        # C. 检查 LaTeX 是否能成功解析出路径
        paths = sub_index._extract_paths(latex)
        if not paths:
            stats["parsing_error"].append(qid)
            continue
        
        # D. 万里挑一：合格的查询
        stats["valid"].append(qid)

    # 3. 输出最终对账报告
    print("\n" + "═"*50)
    print("📊 最终审计报告 (Audit Results)")
    print("═"*50)
    print(f"1. 原始查询总数:          {len(all_queries)}")
    print(f"2. 缺失标注 (Label Missing): {len(stats['missing_label'])} 条")
    print(f"3. 库外真值 (Out of Corpus): {len(stats['out_of_corpus'])} 条")
    print(f"4. 解析失败 (Parsing Error): {len(stats['parsing_error'])} 条")
    print(f"5. 有效评估集 (Valid Set):   {len(stats['valid'])} 条")
    print("═"*50)

    if len(stats['valid']) == 76:
        print("✅ 审计结论：数据完全吻合！分母 76 是严格根据数据一致性得出的。")
    else:
        print(f"⚠️ 审计结论：数据存在偏差！当前有效数为 {len(stats['valid'])}。")

    # 输出具体的缺失 ID 供你人工去 relevance_labels.json 核实
    if stats["missing_label"]:
        print(f"\n💡 你可以去标准答案文件中搜一下这些 ID，应该搜不到: {stats['missing_label'][:5]}...")

if __name__ == "__main__":
    audit()