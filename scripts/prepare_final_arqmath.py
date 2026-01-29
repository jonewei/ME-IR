import json
import csv
import sys
import re
import pickle
from pathlib import Path
from tqdm import tqdm

# 确保导入路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval.approach0_hash import DualHashGenerator, Approach0HashIndex

csv.field_size_limit(sys.maxsize)

# =========================== 核心逻辑：处理查询 TSV ===========================
def process_queries(base_path, hash_gen):
    """直接从官方 TSV 提取 Task 2 的查询公式"""
    tsv_path = base_path / "data" / "arqmath3" / "queries_arqmath3_task2.tsv"
    out_path = base_path / "data" / "processed" / "queries_full.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    
    queries = {}
    print(f"\n🔎 正在从 TSV 提取查询公式...")
    if not tsv_path.exists():
        print(f"⚠️ 警告: 找不到 {tsv_path}，请确认文件路径！")
        return

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                topic_id = row[0].strip()
                raw_latex = row[1].strip()
                # 统一使用 DualHashGenerator 的清洗逻辑
                queries[topic_id] = hash_gen.clean_latex(raw_latex)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print(f"✅ 查询集已就绪: {len(queries)} 条 -> {out_path}")

# =========================== 核心逻辑：语料处理 (Visual ID 对齐) ===========================
def process_corpus(num_shards=101):
    base_path = Path.cwd()
    latex_dir = base_path / "data" / "arqmath3" / "latex_representation_v3"
    
    hash_gen = DualHashGenerator()
    h_index = Approach0HashIndex()
    
    # 1. 先处理查询
    process_queries(base_path, hash_gen)
    
    # 2. 存储元数据：key 必须是 visual_id
    corpus = {} 
    
    latex_files = sorted(latex_dir.glob("*.tsv"))[:num_shards]
    print(f"\n🔄 正在处理 {len(latex_files)} 个语料分片...")
    
    for f in tqdm(latex_files, desc="Processing Shards"):
        with open(f, 'r', encoding='utf-8') as fin:
            reader = csv.reader(fin, delimiter='\t')
            next(reader, None)  # 跳过表头
            
            for row in reader:
                if len(row) < 9: continue
                
                # README 结构: [0:id, 6:visual_id, 7:issue, 8:formula]
                visual_id = row[6].strip()
                issue = row[7].strip()
                raw_latex = row[8].strip()
                
                # 过滤 'd' (不存在于XML)
                if 'd' in issue: continue
                
                # Visual ID 去重 (同一公式只索引一次)
                if visual_id in corpus: continue
                
                clean_norm = hash_gen.clean_latex(raw_latex)
                
                corpus[visual_id] = {
                    "formula_id": visual_id,
                    "latex": raw_latex,
                    "latex_norm": clean_norm
                }
                
                # 构建哈希索引
                h_val = hash_gen.generate_latex_hash(clean_norm)
                if h_val not in h_index.index:
                    h_index.index[h_val] = []
                h_index.index[h_val].append(visual_id)

    # 3. 导出
    out_dir = base_path / "data" / "processed"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n💾 正在导出对齐后的索引数据...")
    with open(out_dir / "formulas.json", 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False)

    h_index.save(base_path / "artifacts" / "approach0_index.pkl")
    
    print(f"✅ 处理完成！")
    print(f"   - 唯一 Visual ID 数量: {len(corpus):,}")
    print(f"   - 语料元数据 -> {out_dir}/formulas.json")
    print(f"   - 哈希索引 -> artifacts/approach0_index.pkl")

if __name__ == "__main__":
    # 执行全流程
    process_corpus(num_shards=101) # 也可以直接改为 101