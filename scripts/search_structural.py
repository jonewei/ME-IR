# # import pickle
# # import json
# # import os
# # import re
# # from collections import defaultdict
# # from tqdm import tqdm

# # # # 配置路径
# # # INDEX_PATH = "data/indices/ipi_index.bin"
# # # QUERY_PATH = "data/processed/queries_full.json"
# # # OUTPUT_PATH = "results/ipi_results.txt"

# # import pickle, json, os, re
# # from collections import defaultdict
# # from tqdm import tqdm

# # def extract_structural_paths(latex):
# #     # 必须与 build 脚本逻辑一致
# #     tokens = re.findall(r'\\[a-zA-Z]+|[\w]+|[{}()^|_=+]', str(latex))
# #     paths = []
# #     stack = []
# #     for t in tokens:
# #         if t == '{': stack.append("sub")
# #         elif t == '}': 
# #             if stack: stack.pop()

# import pickle, json, os, re
# from collections import defaultdict
# from tqdm import tqdm

# def extract_structural_paths(latex):
#     tokens = re.findall(r'\\[a-zA-Z]+|[\w]+|[{}()^|_=+]', str(latex))
#     paths, stack = [], []
#     for t in tokens:
#         if t == '{': stack.append("sub")
#         elif t == '}': 
#             if stack: stack.pop()
#         elif t in ['^', '_']: stack.append("sup" if t == '^' else "sub")
#         else:
#             prefix = "_".join(stack[-2:])
#             path = f"{prefix}_{t}" if prefix else t
#             paths.append(path)
#             if stack and stack[-1] in ["sup", "sub"]: stack.pop()
#     return paths

# def search():
#     print("[*] 正在加载索引...")
#     with open("data/indices/ipi_index.bin", 'rb') as f:
#         index = pickle.load(f)
    
#     with open("data/processed/queries_full.json", 'r') as f:
#         queries = json.load(f)

#     print("[*] 开始加速搜索...")
#     with open("results/ipi_results.txt", 'w') as f_out:
#         for qid, latex in tqdm(queries.items(), desc="Fast Searching"):
#             paths = extract_structural_paths(latex)
            
#             # 提速逻辑：只关注索引中存在的路径
#             valid_paths = [p for p in paths if p in index]
            
#             # 使用局部计分器
#             scores = defaultdict(float)
#             for p in valid_paths:
#                 # 获取该路径下的所有 doc_id 和频次
#                 doc_hits = index[p]
#                 for fid, count in doc_hits.items():
#                     scores[fid] += count
            
#             # 排序并取 Top-1000
#             if scores:
#                 sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
#                 for i, (fid, score) in enumerate(sorted_res):
#                     f_out.write(f"{qid} Q0 {fid} {i+1} {score:.4f} IPI_v3\n")

#     print(f"✅ 检索完成！结果已保存。")

# if __name__ == "__main__":
#     search()

import pickle, json, os, re
from collections import defaultdict
from tqdm import tqdm

def search():
    with open("data/indices/ipi_index.bin", 'rb') as f:
        data = pickle.load(f)
    index = data['index']
    idf_map = data['idf']

    with open("data/processed/queries_full.json", 'r') as f:
        queries = json.load(f)

    with open("results/ipi_results.txt", 'w') as f_out:
        for qid, latex in tqdm(queries.items(), desc="Weighted Searching"):
            paths = [p for p in re.findall(r'\\[a-zA-Z]+|[\w]+|[{}()^|_=+]', str(latex)) if p in index]
            
            scores = defaultdict(float)
            for p in paths:
                weight = idf_map.get(p, 0)
                
                # --- 核心改进：动态权重过滤 ---
                # 过滤掉极其常见的原子符号 (IDF太低的说明它是噪声)
                if weight < 2.0: continue 
                
                # 如果是带结构的路径 (包含 '_')，额外加权
                bonus = 1.5 if '_' in p else 1.0
                
                for fid, tf in index[p].items():
                    # 类似 BM25 的饱和分值，防止长公式霸榜
                    scores[fid] += (tf / (tf + 2.0)) * weight * bonus
            
            if scores:
                sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
                for i, (fid, score) in enumerate(sorted_res):
                    f_out.write(f"{qid} Q0 {fid} {i+1} {score:.4f} IPI_TFIDF\n")

if __name__ == "__main__":
    search()