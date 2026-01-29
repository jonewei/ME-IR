import json
import csv
import sys
import re
import pickle
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# 确保导入路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval.approach0_hash import DualHashGenerator, Approach0HashIndex

csv.field_size_limit(sys.maxsize)

# ===========================  ARQMath 优化版符号映射 ===========================
ARQMATH_SYMBOL_MAPPING = {
    # 分隔符等价
    r'\|': '||',
    r'\Vert': '||',
    r'\lbrace': '{',
    r'\rbrace': '}',
    r'\langle': '<',
    r'\rangle': '>',
    
    # 希腊字母变体（保守合并）
    r'\varepsilon': r'\epsilon',
    r'\varphi': r'\phi',
    # 注意：保留 \vartheta, \varpi 等，它们在某些数学领域有特殊含义
    
    # 关系符号
    r'\le': r'\leq',
    r'\ge': r'\geq',
    r'\ne': r'\neq',
    
    # 箭头符号
    r'\to': r'\rightarrow',
    r'\gets': r'\leftarrow',
    r'\implies': r'\Rightarrow',
    r'\iff': r'\Leftrightarrow',
    
    # 逻辑符号
    r'\land': r'\wedge',
    r'\lor': r'\vee',
    r'\lnot': r'\neg',
    
    # 集合论
    r'\empty': r'\emptyset',
    r'\varnothing': r'\emptyset',
    
    # 转置符号（保守处理）
    r'^\top': '^T',
    r'^t': '^T',
    
    # 省略号
    r'\ldots': r'\cdots',
    r'\dots': r'\cdots',
}

class ARQMathDualHashGenerator(DualHashGenerator):
    """ARQMath-3 特化的哈希生成器"""
    
    def __init__(self):
        super().__init__()
        # 使用 ARQMath 优化的映射表
        self.sorted_symbols = sorted(
            ARQMATH_SYMBOL_MAPPING.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        # 只移除这些字体命令（保留 \mathbb, \mathcal 因为有语义）
        self.font_commands = [
            r'\\mathbf', r'\\mathrm', r'\\mathit', 
            r'\\mathsf', r'\\mathtt', r'\\text', r'\\bm'
        ]
    
    def clean_latex(self, latex_str):
        """ARQMath 优化版清洗"""
        if not latex_str: 
            return "", False
        
        original = latex_str
        
        # 1. 移除定界符
        s = re.sub(r'\$\$?|\\\[|\\\]|\\\(|\\\)', '', latex_str)
        
        # 2. 剥离字体装饰（保守策略）
        for cmd in self.font_commands:
            s = s.replace(cmd, '')
        
        # 3. 符号别名替换
        for old, new in self.sorted_symbols:
            s = s.replace(old, new)
        
        # 4. 统一矩阵环境
        matrix_types = ['pmatrix', 'bmatrix', 'vmatrix', 'Vmatrix']
        for mtype in matrix_types:
            s = re.sub(rf'\\begin\{{{mtype}\}}', r'\\begin{matrix}', s)
            s = re.sub(rf'\\end\{{{mtype}\}}', r'\\end{matrix}', s)
        
        # 5. 移除视觉装饰（保留 \limits，影响语义）
        s = re.sub(r'\\left|\\right|\\displaystyle', '', s)
        
        # 6. 空格标准化（重要：不要完全移除！）
        s = re.sub(r'\s+', ' ', s.strip())
        
        # 7. 简化冗余大括号（仅单字符，保护下标上标）
        s = re.sub(r'\{([a-zA-Z0-9])\}', r'\1', s)
        
        # 判断是否发生实质性改动
        original_normalized = re.sub(r'\s+', ' ', 
                                     re.sub(r'\$\$?|\\\[|\\\]|\\\(|\\\)', '', original)).strip()
        is_normalized = (s != original_normalized)
        
        return s, is_normalized

# =========================== 查询处理 ===========================
def process_queries(base_path, hash_gen):
    """从官方 TSV 提取 Task 2 的查询公式"""
    tsv_path = base_path / "data" / "arqmath3" / "queries_arqmath3_task2.tsv"
    out_path = base_path / "data" / "processed" / "queries_full.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    
    queries = {}
    queries_metadata = {}
    
    print(f"\n🔎 正在从 TSV 提取查询公式...")
    if not tsv_path.exists():
        print(f"⚠️ 警告: 找不到 {tsv_path}")
        return

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                topic_id = row[0].strip()
                raw_latex = row[1].strip()
                
                # 清洗并生成哈希
                clean_latex, is_norm = hash_gen.clean_latex(raw_latex)
                h_latex = hash_gen.generate_latex_hash(clean_latex)
                
                queries[topic_id] = clean_latex
                queries_metadata[topic_id] = {
                    "topic_id": topic_id,
                    "raw_latex": raw_latex,
                    "clean_latex": clean_latex,
                    "hash": h_latex,
                    "is_normalized": is_norm
                }

    # 导出两个文件：简化版和详细版
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    
    with open(out_path.parent / "queries_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(queries_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 查询集已就绪: {len(queries)} 条")
    print(f"   - 简化版 -> {out_path}")
    print(f"   - 详细版 -> {out_path.parent}/queries_metadata.json")

# =========================== 语料处理 (Visual ID 对齐) ===========================
def process_corpus(num_shards=101):
    base_path = Path.cwd()
    latex_dir = base_path / "data" / "arqmath3" / "latex_representation_v3"
    
    hash_gen = ARQMathDualHashGenerator()
    h_index = Approach0HashIndex()
    
    # 1. 先处理查询
    process_queries(base_path, hash_gen)
    
    # 2. 核心数据结构
    corpus = {}  # key: visual_id, value: 公式元数据
    visual_id_stats = defaultdict(int)  # 统计每个 visual_id 出现次数
    issue_stats = defaultdict(int)  # 统计 issue 类型分布
    
    latex_files = sorted(latex_dir.glob("*.tsv"))[:num_shards]
    print(f"\n🔄 正在处理 {len(latex_files)} 个语料分片...")
    
    total_formulas = 0
    skipped_d = 0  # 跳过的 'd' 标记
    skipped_duplicate = 0  # 跳过的重复 visual_id
    
    for f in tqdm(latex_files, desc="Processing Shards"):
        with open(f, 'r', encoding='utf-8') as fin:
            reader = csv.reader(fin, delimiter='\t')
            next(reader, None)  # 跳过表头
            
            for row in reader:
                if len(row) < 9: 
                    continue
                
                total_formulas += 1
                
                # README 字段结构
                formula_id = row[0].strip()
                post_id = row[1].strip()
                thread_id = row[2].strip()
                post_type = row[3].strip()
                comment_id = row[4].strip()
                old_visual_id = row[5].strip()
                visual_id = row[6].strip()
                issue = row[7].strip()
                raw_latex = row[8].strip()
                
                # 统计 issue 分布
                if issue:
                    issue_stats[issue] += 1
                
                # 过滤规则 1: 跳过 'd' 标记（不存在于 XML）
                if 'd' in issue:
                    skipped_d += 1
                    continue
                
                # 过滤规则 2: Visual ID 去重（同一公式只保留一次）
                if visual_id in corpus:
                    skipped_duplicate += 1
                    visual_id_stats[visual_id] += 1
                    continue
                
                visual_id_stats[visual_id] = 1
                
                # 清洗公式
                clean_latex, is_norm = hash_gen.clean_latex(raw_latex)
                h_latex = hash_gen.generate_latex_hash(clean_latex)
                
                # 存储元数据（key 必须是 visual_id！）
                corpus[visual_id] = {
                    "formula_id": formula_id,
                    "visual_id": visual_id,
                    "old_visual_id": old_visual_id,
                    "post_id": post_id,
                    "thread_id": thread_id,
                    "type": post_type,
                    "comment_id": comment_id,
                    "latex": raw_latex,
                    "latex_norm": clean_latex,
                    "hash": h_latex,
                    "is_normalized": is_norm,
                    "issue": issue
                }
                
                # 构建哈希索引（倒排索引：hash -> [visual_ids]）
                if h_latex not in h_index.index:
                    h_index.index[h_latex] = []
                h_index.index[h_latex].append(visual_id)

    # 3. 统计报告
    print(f"\n📊 数据统计:")
    print(f"   - 总公式数: {total_formulas:,}")
    print(f"   - 唯一 Visual ID: {len(corpus):,}")
    print(f"   - 跳过 'd' 标记: {skipped_d:,}")
    print(f"   - 跳过重复 Visual ID: {skipped_duplicate:,}")
    print(f"   - 唯一哈希数: {len(h_index.index):,}")
    print(f"\n📋 Issue 分布:")
    for issue_type, count in sorted(issue_stats.items()):
        print(f"   - '{issue_type}': {count:,}")
    
    # 4. 导出数据
    out_dir = base_path / "data" / "processed"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    artifacts_dir = base_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n💾 正在导出对齐后的索引数据...")
    
    # 导出语料元数据
    with open(out_dir / "formulas.json", 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False)
    
    # 导出哈希索引
    h_index.save(artifacts_dir / "approach0_index.pkl")
    
    # 导出统计信息
    stats = {
        "total_formulas": total_formulas,
        "unique_visual_ids": len(corpus),
        "skipped_d": skipped_d,
        "skipped_duplicate": skipped_duplicate,
        "unique_hashes": len(h_index.index),
        "issue_distribution": dict(issue_stats),
        "visual_id_collision_stats": {
            "max_collision": max(visual_id_stats.values()),
            "avg_collision": sum(visual_id_stats.values()) / len(visual_id_stats),
        }
    }
    
    with open(out_dir / "corpus_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 处理完成！")
    print(f"   - 语料元数据 -> {out_dir}/formulas.json")
    print(f"   - 哈希索引 -> {artifacts_dir}/approach0_index.pkl")
    print(f"   - 统计信息 -> {out_dir}/corpus_stats.json")

# =========================== 检索接口 ===========================
def search_formula(query_latex, index_path, corpus_path, top_k=100):
    """
    ARQMath-3 公式检索接口
    
    Args:
        query_latex: 查询公式的 LaTeX 字符串
        index_path: 哈希索引路径
        corpus_path: 语料元数据路径
        top_k: 返回 top-k 结果
    
    Returns:
        List of (visual_id, score, metadata)
    """
    # 加载索引
    h_index = Approach0HashIndex()
    h_index.load(index_path)
    
    # 加载语料
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    # 清洗查询
    hash_gen = ARQMathDualHashGenerator()
    clean_query, _ = hash_gen.clean_latex(query_latex)
    h_query = hash_gen.generate_latex_hash(clean_query)
    
    # 精确哈希匹配
    visual_ids = h_index.search(h_query)
    
    # 组装结果
    results = []
    for vid in visual_ids[:top_k]:
        if vid in corpus:
            results.append({
                "visual_id": vid,
                "score": 1.0,  # 精确匹配得分
                "metadata": corpus[vid]
            })
    
    return results

def diagnose_index():
    """诊断索引构建是否正确"""
    base_path = Path.cwd()
    
    # 加载索引和语料
    h_index = Approach0HashIndex()
    h_index.load(base_path / "artifacts" / "approach0_index.pkl")
    
    with open(base_path / "data" / "processed" / "formulas.json", 'r') as f:
        corpus = json.load(f)
    
    with open(base_path / "data" / "processed" / "queries_metadata.json", 'r') as f:
        queries = json.load(f)
    
    print("\n🔍 索引诊断报告")
    print("=" * 60)
    
    # 1. 索引基本信息
    print(f"\n1️⃣ 索引统计:")
    print(f"   - 哈希桶数量: {len(h_index.index):,}")
    print(f"   - 语料 Visual ID 数量: {len(corpus):,}")
    
    # 2. 随机抽样检查索引
    hash_gen = ARQMathDualHashGenerator()
    print(f"\n2️⃣ 随机抽样验证 (前 5 个公式):")
    
    for i, (visual_id, metadata) in enumerate(list(corpus.items())[:5], 1):
        raw_latex = metadata['latex']
        clean_latex = metadata['latex_norm']
        stored_hash = metadata['hash']
        
        # 重新计算哈希
        recalc_clean, _ = hash_gen.clean_latex(raw_latex)
        recalc_hash = hash_gen.generate_latex_hash(recalc_clean)
        
        print(f"\n   样本 {i}:")
        print(f"   - Visual ID: {visual_id}")
        print(f"   - 原始 LaTeX: {raw_latex[:60]}...")
        print(f"   - 清洗后: {clean_latex[:60]}...")
        print(f"   - 存储哈希: {stored_hash[:16]}...")
        print(f"   - 重算哈希: {recalc_hash[:16]}...")
        print(f"   - 哈希一致: {'✅' if stored_hash == recalc_hash else '❌'}")
        
        # 检查索引中是否能找到
        if recalc_hash in h_index.index:
            found_vids = h_index.index[recalc_hash]
            print(f"   - 索引查找: ✅ 找到 {len(found_vids)} 个匹配")
            print(f"   - 本 Visual ID 在结果中: {'✅' if visual_id in found_vids else '❌'}")
        else:
            print(f"   - 索引查找: ❌ 未找到")
    
    # 3. 查询集检查
    print(f"\n3️⃣ 查询集验证 (前 3 个查询):")
    
    query_found = 0
    for i, (topic_id, query_meta) in enumerate(list(queries.items())[:3], 1):
        query_hash = query_meta['hash']
        query_latex = query_meta['raw_latex']
        
        print(f"\n   查询 {i} (Topic {topic_id}):")
        print(f"   - 原始 LaTeX: {query_latex[:60]}...")
        print(f"   - 清洗后: {query_meta['clean_latex'][:60]}...")
        print(f"   - 查询哈希: {query_hash[:16]}...")
        
        if query_hash in h_index.index:
            matches = h_index.index[query_hash]
            print(f"   - 匹配结果: ✅ 找到 {len(matches)} 个 Visual ID")
            query_found += 1
            # 显示前 3 个匹配
            for vid in matches[:3]:
                if vid in corpus:
                    print(f"     - {vid}: {corpus[vid]['latex'][:50]}...")
        else:
            print(f"   - 匹配结果: ❌ 未找到")
    
    print(f"\n4️⃣ 查询覆盖率:")
    print(f"   - 前 3 个查询中有匹配: {query_found}/3")
    
    # 4. 哈希冲突分析
    collision_counts = [len(vids) for vids in h_index.index.values()]
    print(f"\n5️⃣ 哈希冲突统计:")
    print(f"   - 平均每个哈希对应 Visual ID 数: {sum(collision_counts) / len(collision_counts):.2f}")
    print(f"   - 最大冲突数: {max(collision_counts)}")
    print(f"   - 单一映射比例: {sum(1 for c in collision_counts if c == 1) / len(collision_counts) * 100:.2f}%")
    
    # 5. 验证 Visual ID 唯一性
    all_vids_in_index = set()
    for vids in h_index.index.values():
        all_vids_in_index.update(vids)
    
    print(f"\n6️⃣ Visual ID 完整性:")
    print(f"   - 语料中的 Visual ID: {len(corpus):,}")
    print(f"   - 索引中的 Visual ID: {len(all_vids_in_index):,}")
    print(f"   - 覆盖率: {len(all_vids_in_index) / len(corpus) * 100:.2f}%")
    
    missing_vids = set(corpus.keys()) - all_vids_in_index
    if missing_vids:
        print(f"   - ⚠️ 有 {len(missing_vids)} 个 Visual ID 未在索引中")
        print(f"   - 示例: {list(missing_vids)[:3]}")

def test_retrieval():
    """使用实际存在的查询进行测试"""
    base_path = Path.cwd()
    
    # 加载查询集
    with open(base_path / "data" / "processed" / "queries_metadata.json", 'r') as f:
        queries = json.load(f)
    
    print("\n🧪 测试检索功能")
    print("=" * 60)
    
    # 测试前 5 个查询
    test_count = 0
    found_count = 0
    
    for topic_id, query_meta in list(queries.items())[:5]:
        test_count += 1
        query_latex = query_meta['raw_latex']
        
        print(f"\n测试 {test_count}: Topic {topic_id}")
        print(f"查询公式: {query_latex[:80]}...")
        
        results = search_formula(
            query_latex=query_latex,
            index_path=base_path / "artifacts" / "approach0_index.pkl",
            corpus_path=base_path / "data" / "processed" / "formulas.json",
            top_k=10
        )
        
        if results:
            found_count += 1
            print(f"✅ 找到 {len(results)} 条匹配")
            
            # 显示前 3 个结果
            for i, r in enumerate(results[:3], 1):
                print(f"   {i}. Visual ID: {r['visual_id']}")
                print(f"      LaTeX: {r['metadata']['latex'][:60]}...")
                print(f"      Post ID: {r['metadata']['post_id']}")
        else:
            print(f"❌ 未找到匹配")
    
    print(f"\n📊 测试总结:")
    print(f"   - 测试查询数: {test_count}")
    print(f"   - 成功找到结果: {found_count}")
    print(f"   - 成功率: {found_count / test_count * 100:.1f}%")
    
    if found_count == 0:
        print(f"\n⚠️ 警告: 所有测试查询都未找到结果")
        print(f"   可能的原因:")
        print(f"   1. 只处理了 10 个分片，查询公式可能在其他分片中")
        print(f"   2. 符号标准化策略导致查询和语料的哈希不匹配")
        print(f"   3. 查询公式在原始数据集中就不存在")



if __name__ == "__main__":
    import sys
    
    # 步骤 1: 处理语料（测试用 10 分片）
    print("步骤 1: 处理 ARQMath-3 语料")
    print("=" * 60)
    process_corpus(num_shards=10)
    
    # 步骤 2: 诊断索引
    print("\n\n步骤 2: 诊断索引构建")
    print("=" * 60)
    diagnose_index()
    
    # 步骤 3: 测试检索
    print("\n\n步骤 3: 测试检索功能")
    print("=" * 60)
    test_retrieval()
    
    # 步骤 4: 建议
    print("\n\n步骤 4: 下一步建议")
    print("=" * 60)
    print("如果检索测试成功率较低，可能需要:")
    print("1. 处理完整的 101 个分片以获得完整覆盖")
    print("2. 调整符号标准化策略（ARQMATH_SYMBOL_MAPPING）")
    print("3. 添加更多检索级别（结构匹配、软匹配等）")
    print("\n运行完整版本:")
    print("   process_corpus(num_shards=101)")