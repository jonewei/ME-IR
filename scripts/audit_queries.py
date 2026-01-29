import json
import csv

def run_audit():
    print("--- LS-MIR 数据治理审计开始 ---")

    # A. 读取初始题库 (100个)
    with open("data/processed/queries_full.json", 'r', encoding='utf-8') as f:
        all_queries = json.load(f)
    
    # 自动适配 JSON 结构
    if isinstance(all_queries, dict):
        # 如果是 {"B.1": {...}, "B.2": {...}} 结构
        initial_ids = set(all_queries.keys())
    elif isinstance(all_queries, list):
        # 如果是 ["B.1", "B.2"] 或 [{"visual_id": "B.1"}, ...] 结构
        if len(all_queries) > 0 and isinstance(all_queries[0], dict):
            initial_ids = set([q.get('visual_id', q.get('id')) for q in all_queries])
        else:
            initial_ids = set(all_queries)
    
    print(f"[1] 初始定义的查询总数: {len(initial_ids)}")

    # B. 读取官方提供的标准答案 (Qrels)
    official_valid_ids = set()
    with open("data/arqmath3/qrel_task2_2022_official.tsv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row: official_valid_ids.add(row[0])
    print(f"[2] 官方真值集中具备答案的查询数: {len(official_valid_ids)}")

    # C. 读取最终实验用的 76 个专家集
    with open("data/qrel_76_expert.json", 'r', encoding='utf-8') as f:
        expert_data = json.load(f)
    expert_ids = set(expert_data.keys())
    print(f"[3] 最终参与实验评估的查询数: {len(expert_ids)}")

    print("\n" + "="*40)
    print(f"{'审计项目':<15} | {'数值':<10}")
    print("-" * 30)
    
    no_labels = initial_ids - official_valid_ids
    parsing_failed = official_valid_ids - expert_ids
    
    print(f"{'1. 标注稀疏剔除':<15} | -{len(no_labels)}")
    print(f"{'2. 解析合法性剔除':<15} | -{len(parsing_failed)}")
    print(f"{'最终有效测试集':<15} | {len(expert_ids)}")
    print("="*40)

if __name__ == "__main__":
    run_audit()