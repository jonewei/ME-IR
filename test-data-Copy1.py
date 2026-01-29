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


# 伪代码：核实你的 76 是怎么来的
all_topics = set(range(1, 101))  # 假设是 Topic 1-100
with open("qrels.txt", "r") as f:
    qrels_topics = set([line.split()[0] for line in f]) # 官方有标注的 Topic

# 1. 算出真值缺失的数量
missing_qrels = all_topics - qrels_topics 
print(f"真值可用性过滤: -{len(missing_qrels)}")

# 2. 算出解析失败的数量
# 检查你的输出文件（如 result.trec），看哪些 Topic 消失了
final_results_topics = set(your_result_ids)
parsing_failed = qrels_topics - final_results_topics
print(f"解析合法性过滤: -{len(parsing_failed)}")
