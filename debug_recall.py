import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from retrieval.approach0_hash import skeleton_hash

# 1. 加载数据
with open('data/processed/formulas.json') as f: formulas = json.load(f)
with open('data/processed/queries_full.json') as f: queries_full = json.load(f)
with open('data/processed/relevance_labels.json') as f: labels = json.load(f)

qid = "B.301"
q_data = queries_full[qid]
q_latex = q_data['latex']
q_mathml = q_data.get('mathml_skel', "")

# ✅ 核心：生成查询的 MathML 哈希
if q_mathml:
    q_hash = skeleton_hash("", mathml_skel=q_mathml)
    print(f"=== 深度追踪 (MathML 模式): {qid} ===")
    print(f"Query MathML Skel: {q_mathml[:100]}...")
else:
    q_hash = skeleton_hash(q_latex)
    print(f"=== 深度追踪 (LaTeX 模式): {qid} ===")

gt_ids = list(labels.get(qid, {}).keys())
match_count = 0
found_in_corpus = 0

for fid in gt_ids:
    if fid in formulas:
        found_in_corpus += 1
        f_data = formulas[fid]
        
        # 库里存了两个哈希，我们检查查询哈希是否命中其中任何一个
        t_hash_latex = skeleton_hash(f_data.get('latex', ''))
        t_hash_mathml = skeleton_hash("", mathml_skel=f_data.get('mathml_skel', ''))
        
        if q_hash == t_hash_latex or q_hash == t_hash_mathml:
            match_count += 1
            print(f"✅ ID {fid:10s}: [MATCH] 结构完美对齐！")

print(f"\n" + "="*30)
print(f"结论：库里有 {found_in_corpus}/{len(gt_ids)} 个答案。")
print(f"使用 MathML 语义对齐后，直接命中了 {match_count} 个。")
print("="*30)
