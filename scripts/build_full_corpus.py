import json
import csv
import sys
import pickle
from pathlib import Path
from tqdm import tqdm

# 强制引入
sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval.approach0_hash import DualHashGenerator, Approach0HashIndex

csv.field_size_limit(sys.maxsize)

def build_full_system():
    base_path = Path.cwd()
    latex_dir = base_path / "data" / "arqmath3" / "latex_representation_v3"
    out_dir = base_path / "data" / "processed"
    artifact_dir = base_path / "artifacts"
    
    out_dir.mkdir(exist_ok=True, parents=True)
    artifact_dir.mkdir(exist_ok=True)

    hash_gen = DualHashGenerator()
    
    # --- Part 1: 处理查询 ---
    print("🔎 正在提取查询集...")
    queries = {}
    q_tsv = base_path / "data" / "arqmath3" / "queries_arqmath3_task2.tsv"
    with open(q_tsv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                q_norm, _ = hash_gen.clean_latex(row[1].strip())
                queries[row[0].strip()] = q_norm
    
    query_file = out_dir / "queries_full.json"
    with open(query_file, 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    # --- Part 2: 处理语料 ---
    # 自动识别目录下所有的 TSV 分片
    all_shards = sorted(list(latex_dir.glob("*.tsv")))
    corpus = {}
    h_index = Approach0HashIndex()
    
    # 详细统计指标
    stats = {
        "total_instances": 0,    # 读取的总行数
        "skipped_issue_d": 0,   # 官方标记无效的
        "duplicate_skipped": 0, # 因 Visual ID 重复而跳过的
        "unique_visual_ids": 0, # 最终入库的唯一 ID
        "normalized_count": 0   # 触发增强清洗规则的
    }

    print(f"🚀 启动扫描。发现分片总数: {len(all_shards)}")
    # 如果想跑全量，不要切片；如果想先测试，可以用 all_shards[:101]
    for shard in tqdm(all_shards, desc="Processing Shards"):
        with open(shard, 'r', encoding='utf-8') as fin:
            reader = csv.reader(fin, delimiter='\t')
            next(reader, None) # 跳过表头
            for row in reader:
                if len(row) < 9: continue
                stats["total_instances"] += 1
                
                visual_id = row[6].strip()
                issue = row[7].strip()
                raw_latex = row[8].strip()
                
                # 过滤逻辑 1: 无效公式
                if 'd' in issue:
                    stats["skipped_issue_d"] += 1
                    continue
                
                # 过滤逻辑 2: 重复 Visual ID (核心去重点)
                if visual_id in corpus:
                    stats["duplicate_skipped"] += 1
                    continue
                
                # 执行清洗
                norm_latex, was_norm = hash_gen.clean_latex(raw_latex)
                if was_norm: stats["normalized_count"] += 1
                
                # 入库
                corpus[visual_id] = {
                    "formula_id": visual_id,
                    "latex": raw_latex,
                    "latex_norm": norm_latex
                }
                
                # 索引哈希
                h_val = hash_gen.generate_latex_hash(norm_latex)
                if h_val not in h_index.index:
                    h_index.index[h_val] = []
                h_index.index[h_val].append(visual_id)
                
                stats["unique_visual_ids"] += 1

    # --- Part 3: 保存与汇总 ---
    print("\n💾 正在保存索引文件...")
    corpus_file = out_dir / "formulas.json"
    index_file = artifact_dir / "approach0_index.pkl"
    
    with open(corpus_file, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False)
    h_index.save(index_file)
    
    print("\n" + "="*50)
    print("📊 最终构建汇总报告")
    print("="*50)
    print(f"1. 原始实例总数 (Instances):   {stats['total_instances']:,}")
    print(f"2. 无效数据过滤 (Issue 'd'):  {stats['skipped_issue_d']:,}")
    print(f"3. 重复公式过滤 (Duplicates): {stats['duplicate_skipped']:,}")
    print(f"4. 唯一 Visual ID (Index Size): {stats['unique_visual_ids']:,}")
    print(f"   (去重率: {stats['duplicate_skipped']/max(1, stats['total_instances'])*100:.2f}%)")
    print(f"5. 符号规范化命中次数:         {stats['normalized_count']:,}")
    print("-" * 50)
    print("📁 已生成文件清单:")
    print(f"   - 查询集 JSON:  {query_file}")
    print(f"   - 语料元数据:   {corpus_file}")
    print(f"   - 哈希路索引:   {index_file}")
    print("="*50)
    print("💡 提示：如果唯一 ID 数远低于 2800 万，请确认是否扫描了全部 300+ 个分片。")

if __name__ == "__main__":
    build_full_system()