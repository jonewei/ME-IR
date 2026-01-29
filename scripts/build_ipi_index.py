import pickle
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# 配置路径
# 假设你的语料库路径
CORPUS_PATH = "data/processed/formulas.json"
INDEX_OUTPUT = "data/indices/ipi_index.bin"

def extract_structural_paths(latex):
    """
    LS-MIR 核心算法：提取结构路径
    这里实现一个基于符号深度的简化版拓扑提取逻辑
    """
    tokens = latex.replace('{', ' { ').replace('}', ' } ').replace('\\', ' \\').split()
    paths = []
    stack = []
    
    for t in tokens:
        if t == '{':
            stack.append("sub")
        elif t == '}':
            if stack: stack.pop()
        else:
            # 构造路径：层级 + 符号 (例如: sub_sub_alpha)
            path = "_".join(stack + [t])
            if len(t) > 1 or t.isalpha(): # 过滤掉简单的括号和单字符
                paths.append(path)
    return paths

def build_structural_index():
    print(f"[*] 开始构建 IPI 结构索引 (目标: 8.41M)...")
    
    # 倒排表结构: { path_string: { formula_id: count } }
    inverted_index = defaultdict(lambda: defaultdict(int))
    
    if not os.path.exists("data/indices"):
        os.makedirs("data/indices")

    # 分块读取语料库
    reader = pd.read_csv(
        CORPUS_PATH, 
        sep='\t', 
        names=['id', 'latex'], 
        quoting=3, 
        chunksize=100000
    )

    total_count = 0
    for chunk in tqdm(reader, desc="Building IPI Index"):
        for _, row in chunk.iterrows():
            fid = str(row['id'])
            latex = str(row['latex'])
            
            # 提取结构特征
            paths = extract_structural_paths(latex)
            
            for p in paths:
                inverted_index[p][fid] += 1
            
            total_count += 1

    print(f"[*] 索引构建完成，正在持久化到 {INDEX_OUTPUT}...")
    # 转换为普通字典以减小 pickle 序列化负担
    final_dict = {k: dict(v) for k, v in inverted_index.items()}
    
    with open(INDEX_OUTPUT, 'wb') as f:
        pickle.dump(final_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"✅ 成功！IPI 索引已保存，覆盖路径数: {len(final_dict)}")

if __name__ == "__main__":
    build_structural_index()