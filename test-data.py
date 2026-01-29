# import json

# # 1. 检查真值表里的 ID
# with open('data/qrel_76_expert.json', 'r') as f:
#     qrel = json.load(f)
#     print(f"真值表 (QREL) 示例 QID: {list(qrel.keys())[:3]}")

# # 2. 检查结果表里的 ID
# with open('results/raw_sem_scores.json', 'r') as f:
#     res = json.load(f)
#     print(f"检索结果 (Results) 示例 QID: {list(res.keys())[:3]}")

# import json

# # 1. 加载真值表里【被标记为相关】的所有公式 ID
# with open('data/qrel_76_expert.json', 'r') as f:
#     qrel = json.load(f)
#     # 提取所有相关的 doc_id
#     relevant_ids = set()
#     for qid, docs in qrel.items():
#         for doc_id, rel in docs.items():
#             if rel > 0:
#                 relevant_ids.add(str(doc_id))
#     print(f"真值表中的相关公式 ID 示例: {list(relevant_ids)[:5]}")
#     print(f"真值表中总共有 {len(relevant_ids)} 个相关的公式 ID")

# # 2. 加载你的语义检索结果中的公式 ID
# with open('results/raw_sem_scores.json', 'r') as f:
#     sem_res = json.load(f)
#     found_ids = set()
#     for qid, docs in sem_res.items():
#         for doc_id in docs.keys():
#             found_ids.add(str(doc_id))
#     print(f"你的结果中的公式 ID 示例: {list(found_ids)[:5]}")

# # 3. 计算交集
# intersection = relevant_ids.intersection(found_ids)
# print(f"\n匹配成功的 ID 数量: {len(intersection)}")
# if len(intersection) == 0:
#     print("❌ 警告：你的结果 ID 与真值表 ID 完全没有交集！")

# import glob

# # 检查一个分片
# shard_file = "data/arqmath3/latex_representation_v3/1.tsv"
# with open(shard_file, 'r') as f:
#     line = f.readlines()[1] # 跳过表头
#     parts = line.strip().split('\t')
#     print(f"列 0 (目前可能用的): {parts[0]}")
#     print(f"列 6 (真值表想要的): {parts[6]}")



# import json
# from collections import defaultdict

# # 路径配置
# QREL_PATH = "data/qrel_76_expert.json"
# SEM_PATH = "results/raw_sem_scores.json"
# FUSED_PATH = "results/raw_str_scores.json" # 这里建议加载融合后的结果，如果没有，我们直接对比语义和结构
# QUERY_TEXT_PATH = "data/processed/queries_full.json"

# def get_rank(run_dict, qid, target_fids):
#     """获取第一个相关文档的排名 (1-based)"""
#     if qid not in run_dict: return 999
#     # 按得分从高到低排序
#     sorted_docs = sorted(run_dict[qid].items(), key=lambda x: x[1], reverse=True)
#     for rank, (fid, _) in enumerate(sorted_docs):
#         if fid in target_fids:
#             return rank + 1
#     return 999

# def find_cases():
#     with open(QREL_PATH, 'r') as f: qrels = json.load(f)
#     with open(SEM_PATH, 'r') as f: sem_run = json.load(f)
#     with open(QUERY_TEXT_PATH, 'r') as f: queries = json.load(f)
    
#     # 我们这里对比语义流 (S1) 和 结构流 (S2)
#     # 或者如果你有融合后的 S4 JSON，效果更好
#     with open(SEM_PATH, 'r') as f: sem_run = json.load(f)
#     # 临时模拟 LS-MIR 逻辑：这里我们直接载入结构流看它的“神来之笔”
#     with open("results/raw_str_scores.json", 'r') as f: str_run = json.load(f)

#     print(f"{'QID':<10} | {'Sem Rank':<10} | {'Str Rank':<10} | {'Improvement'}")
#     print("-" * 50)

#     findings = []
#     for qid, target_docs in qrels.items():
#         relevant_fids = [fid for fid, rel in target_docs.items() if rel > 0]
#         if not relevant_fids: continue

#         rank_sem = get_rank(sem_run, qid, relevant_fids)
#         rank_str = get_rank(str_run, qid, relevant_fids)

#         # 筛选条件：结构流排名远高于语义流（语义没搜到，但结构搜到了）
#         if rank_str < rank_sem and rank_str < 10:
#             findings.append({
#                 'qid': qid,
#                 'sem_rank': rank_sem,
#                 'str_rank': rank_str,
#                 'gain': rank_sem - rank_str,
#                 'latex': queries.get(qid, "N/A")
#             })

#     # 按提升幅度排序
#     findings.sort(key=lambda x: x['gain'], reverse=True)

#     for case in findings[:5]:
#         print(f"{case['qid']:<10} | {case['sem_rank']:<10} | {case['str_rank']:<10} | +{case['gain']}")
#         print(f"Query LaTeX: {case['latex']}\n")

# if __name__ == "__main__":
#     find_cases()


# # 伪代码：核实你的 76 是怎么来的
# all_topics = set(range(1, 101))  # 假设是 Topic 1-100
# with open("qrels.txt", "r") as f:
#     qrels_topics = set([line.split()[0] for line in f]) # 官方有标注的 Topic

# # 1. 算出真值缺失的数量
# missing_qrels = all_topics - qrels_topics 
# print(f"真值可用性过滤: -{len(missing_qrels)}")

# # 2. 算出解析失败的数量
# # 检查你的输出文件（如 result.trec），看哪些 Topic 消失了
# final_results_topics = set(your_result_ids)
# parsing_failed = qrels_topics - final_results_topics
# print(f"解析合法性过滤: -{len(parsing_failed)}")

import json

# 加载数据
with open("data/qrel_76_expert.json", "r") as f:
    qrel = json.load(f)

with open("results/raw_sem_scores.json", "r") as f:
    sem_scores = json.load(f)

# ========== 智能检测数据结构 ==========
print("=" * 60)
print("📂 数据结构分析")
print("=" * 60)

# 检测 QREL 格式
print("\n1️⃣ QREL 数据格式:")
print(f"   类型: {type(qrel)}")
print(f"   总查询数: {len(qrel)}")

# 取第一个查询
first_qid = list(qrel.keys())[0]
first_qrel_value = qrel[first_qid]

print(f"\n   第一个查询 ID: {first_qid}")
print(f"   对应值的类型: {type(first_qrel_value)}")

# 根据类型打印
if isinstance(first_qrel_value, dict):
    print(f"   真值数量: {len(first_qrel_value)}")
    print(f"   前3个真值:")
    for i, (fid, relevance) in enumerate(list(first_qrel_value.items())[:3]):
        print(f"      {fid}: {relevance}")
elif isinstance(first_qrel_value, list):
    print(f"   真值数量: {len(first_qrel_value)}")
    print(f"   前3个真值: {first_qrel_value[:3]}")
else:
    print(f"   原始值: {first_qrel_value}")

# ========== 检测语义分数格式 ==========
print("\n" + "=" * 60)
print("2️⃣ 语义分数数据格式:")
print(f"   类型: {type(sem_scores)}")
print(f"   总查询数: {len(sem_scores)}")

first_score_qid = list(sem_scores.keys())[0]
first_score_value = sem_scores[first_score_qid]

print(f"\n   第一个查询 ID: {first_score_qid}")
print(f"   对应值的类型: {type(first_score_value)}")

if isinstance(first_score_value, dict):
    print(f"   候选数量: {len(first_score_value)}")
    print(f"   前5个候选:")
    for i, (fid, score) in enumerate(list(first_score_value.items())[:5]):
        print(f"      {fid}: {score:.6f}")
elif isinstance(first_score_value, list):
    print(f"   候选数量: {len(first_score_value)}")
    print(f"   前5个候选: {first_score_value[:5]}")

# ========== 交叉验证 ==========
print("\n" + "=" * 60)
print("3️⃣ 数据一致性检查:")

# 检查查询 ID 是否匹配
qrel_ids = set(qrel.keys())
score_ids = set(sem_scores.keys())
common_ids = qrel_ids & score_ids

print(f"   QREL 查询数: {len(qrel_ids)}")
print(f"   分数查询数: {len(score_ids)}")
print(f"   共同查询数: {len(common_ids)}")

if len(common_ids) > 0:
    # 随机选一个共同查询
    sample_qid = list(common_ids)[0]
    print(f"\n   ✅ 示例查询: {sample_qid}")
    
    # 获取真值
    if isinstance(qrel[sample_qid], dict):
        truth_ids = set(qrel[sample_qid].keys())
    else:
        truth_ids = set(qrel[sample_qid])
    
    # 获取候选
    if isinstance(sem_scores[sample_qid], dict):
        candidate_ids = set(sem_scores[sample_qid].keys())
    else:
        candidate_ids = set([x[0] for x in sem_scores[sample_qid]])
    
    # 检查真值是否在候选中
    truth_in_candidates = truth_ids & candidate_ids
    
    print(f"   真值数量: {len(truth_ids)}")
    print(f"   候选数量: {len(candidate_ids)}")
    print(f"   真值在候选中: {len(truth_in_candidates)}/{len(truth_ids)}")
    
    if len(truth_in_candidates) > 0:
        sample_truth = list(truth_in_candidates)[0]
        if isinstance(sem_scores[sample_qid], dict):
            truth_score = sem_scores[sample_qid][sample_truth]
            print(f"   真值 {sample_truth} 的分数: {truth_score:.6f}")

print("\n" + "=" * 60)
print("✅ 数据检测完成！请将上述输出发给我。")
print("=" * 60)

