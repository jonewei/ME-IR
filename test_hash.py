# from retrieval.approach0_hash import skeleton_hash

# # 测试相同输入是否产生相同哈希
# test_skel = "mfrac,mn,msqrt,mi"
# h1 = skeleton_hash("", mathml_skel=test_skel)
# h2 = skeleton_hash("", mathml_skel=test_skel)

# print(f"Skeleton: {test_skel}")
# print(f"Hash 1: {h1}")
# print(f"Hash 2: {h2}")
# print(f"相同? {h1 == h2}")

"""
终极哈希匹配诊断
"""
import json
import pickle
from retrieval.approach0_hash import skeleton_hash

# 1. 加载查询
with open('data/processed/queries_full.json') as f:
    queries = json.load(f)

# 2. 加载索引
with open('artifacts/approach0_index.pkl', 'rb') as f:
    idx_data = pickle.load(f)
    index = idx_data['index']
    all_formulas = idx_data.get('all_formulas', [])

# 3. 测试第一个查询
sample_qid = 'A.301'
query = queries[sample_qid]

print("=" * 80)
print("🔍 哈希匹配终极诊断")
print("=" * 80)

print(f"\n【查询信息】")
print(f"Query ID: {sample_qid}")
print(f"LaTeX: {query['latex'][:80]}...")
print(f"MathML Skeleton: {query['mathml_skel'][:80]}...")

# 计算查询哈希（两种方式）
query_hash_latex = skeleton_hash(query['latex'])
query_hash_mathml = skeleton_hash("", mathml_skel=query['mathml_skel'])

print(f"\n【查询哈希值】")
print(f"LaTeX 哈希: {query_hash_latex}")
print(f"MathML 哈希: {query_hash_mathml}")

# 检查是否在索引中
print(f"\n【索引匹配结果】")
if query_hash_latex in index:
    print(f"✅ LaTeX 哈希在索引中！桶大小: {len(index[query_hash_latex])}")
else:
    print(f"❌ LaTeX 哈希不在索引中")

if query_hash_mathml in index:
    print(f"✅ MathML 哈希在索引中！桶大小: {len(index[query_hash_mathml])}")
else:
    print(f"❌ MathML 哈希不在索引中")

# 4. 检查语料库样本
print(f"\n【语料库样本（前3个）】")
for i, formula in enumerate(all_formulas[:3]):
    fid = formula.get('formula_id', 'N/A')
    latex = formula.get('latex', '')[:50]
    mathml_skel = formula.get('mathml_skel', '')[:50]
    
    # 计算语料库哈希
    corpus_hash_latex = skeleton_hash(latex)
    corpus_hash_mathml = skeleton_hash("", mathml_skel=mathml_skel)
    
    print(f"\n{i+1}. Formula ID: {fid}")
    print(f"   LaTeX: {latex}...")
    print(f"   MathML Skel: {mathml_skel}...")
    print(f"   LaTeX Hash: {corpus_hash_latex}")
    print(f"   MathML Hash: {corpus_hash_mathml}")

# 5. 检查索引样本哈希
print(f"\n【索引样本哈希（前10个）】")
for i, (h, bucket) in enumerate(list(index.items())[:10]):
    print(f"{i+1}. Hash: {h}, Bucket size: {len(bucket)}")

print("\n" + "=" * 80)
