# import pickle, os, re, glob
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

# def build():
#     print("[*] 正在构建索引，ID 锁定为列 6 (visual_id)...")
#     shard_files = sorted(glob.glob("data/arqmath3/latex_representation_v3/*.tsv"))
#     inverted_index = defaultdict(lambda: defaultdict(int))
    
#     for shard_path in tqdm(shard_files, desc="Processing Shards"):
#         with open(shard_path, 'r', encoding='utf-8', errors='ignore') as f:
#             for line in f:
#                 parts = line.strip().split('\t')
#                 # 避开表头，确保取列 6
#                 if len(parts) <= 8 or parts[0] == 'id': continue
                
#                 fid = parts[6]    # 核心修改：使用 visual_id
#                 latex = parts[8]
#                 for p in extract_structural_paths(latex):
#                     inverted_index[p][fid] += 1

#     with open("data/indices/ipi_index.bin", 'wb') as f:
#         pickle.dump(dict(inverted_index), f, protocol=pickle.HIGHEST_PROTOCOL)
#     print(f"✅ 索引构建完成！当前 ID 示例: {list(list(inverted_index.values())[0].keys())[0]}")

# if __name__ == "__main__":
#     build()

import pickle, os, re, glob, math
from collections import defaultdict
from tqdm import tqdm

def extract_structural_paths(latex):
    # 1. 基础分词
    tokens = re.findall(r'\\[a-zA-Z]+|[\w]+|[{}()^|_=+]', str(latex))
    
    paths = []
    stack = []
    for t in tokens:
        if t == '{':
            stack.append("sub")
        elif t == '}': 
            if stack: stack.pop()
        elif t in ['^', '_']:
            stack.append("sup" if t == '^' else "sub")
        else:
            # --- 核心改进：提取 Bigram 结构 ---
            # 当前符号 t
            # 父层级为 stack[-1] 如果存在
            curr_layer = stack[-1] if stack else "root"
            
            # 路径 1: 原子符号 (e.g., "alpha")
            paths.append(t) 
            
            # 路径 2: 父子绑定关系 (e.g., "sub_alpha") - 极具辨识度
            if stack:
                paths.append(f"{stack[-1]}_{t}")
                
            # 路径 3: 深度路径 (e.g., "root_sub_alpha")
            if len(stack) >= 1:
                deep_path = "_".join(stack + [t])
                paths.append(deep_path)

            if stack and stack[-1] in ["sup", "sub"]:
                stack.pop()
    return paths

def build():
    shard_files = sorted(glob.glob("data/arqmath3/latex_representation_v3/*.tsv"))
    # { path: { doc_id: term_frequency } }
    inverted_index = defaultdict(lambda: defaultdict(int))
    # { path: document_frequency }
    df_counts = defaultdict(int)
    total_docs = 0

    for shard_path in tqdm(shard_files, desc="Building TF-IDF Index"):
        with open(shard_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) <= 8 or parts[0] == 'id': continue
                fid, latex = parts[6], parts[8]
                
                paths = extract_structural_paths(latex)
                if not paths: continue
                
                total_docs += 1
                unique_paths_in_doc = set()
                for p in paths:
                    inverted_index[p][fid] += 1
                    unique_paths_in_doc.add(p)
                
                for p in unique_paths_in_doc:
                    df_counts[p] += 1

    # 计算 IDF 并过滤极其常见的噪声 (如出现在 >50% 的公式中的符号)
    idf_map = {}
    for p, df in df_counts.items():
        # 标准 IDF 公式
        idf_map[p] = math.log(total_docs / (1 + df))

    print(f"[*] 索引构建完成。总文档数: {total_docs}, 词项数: {len(idf_map)}")
    
    with open("data/indices/ipi_index.bin", 'wb') as f:
        # 保存索引、IDF 和总文档数
        data = {
            'index': dict(inverted_index),
            'idf': idf_map,
            'total_docs': total_docs
        }
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    build()