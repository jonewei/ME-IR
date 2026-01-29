import pandas as pd
from tqdm import tqdm
import os

# 配置路径 - 请确保与你构建索引时使用的语料库一致
CORPUS_PATH = "data/processed/unique_formulas.tsv"
ID_MAP_OUTPUT = "artifacts/vector_id_map_v4.txt"

def recover_id_map():
    print(f"[*] 正在从语料库恢复 ID 映射关系...")
    
    if not os.path.exists(CORPUS_PATH):
        print(f"❌ 错误: 找不到语料库文件 {CORPUS_PATH}")
        return

    os.makedirs(os.path.dirname(ID_MAP_OUTPUT), exist_ok=True)

    # 逻辑：按行读取，提取 ID。
    # 警告：如果你在构建索引时对数据进行了 shuffle（打乱），此方法将失效！
    # 但标准的 build_vector_index 脚本通常是顺序读取的。
    
    count = 0
    with open(ID_MAP_OUTPUT, 'w', encoding='utf-8') as f_out:
        # 使用 chunksize 避免内存溢出
        reader = pd.read_csv(
            CORPUS_PATH, 
            sep='\t', 
            names=['id', 'latex'], 
            quoting=3, # 这里的参数必须和 build 索引时一致
            chunksize=100000
        )
        
        for chunk in tqdm(reader, desc="Extracting IDs"):
            ids = chunk['id'].astype(str).tolist()
            for fid in ids:
                f_out.write(fid + '\n')
                count += 1

    print(f"✅ 成功恢复 {count} 条 ID 映射，已保存至 {ID_MAP_OUTPUT}")

if __name__ == "__main__":
    recover_id_map()