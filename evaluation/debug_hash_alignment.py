import json
import re
from pathlib import Path
from retrieval.approach0_hash import DualHashGenerator

# =========================== 必须与 prepare 脚本完全一致的清洗函数 ===========================
def clean_latex(latex_str):
    if not latex_str: return ""
    latex_str = re.sub(r'\$\$?|\\\[|\\\]', '', latex_str)
    latex_str = re.sub(r'\\dfrac|\\tfrac', r'\\frac', latex_str)
    latex_str = re.sub(r'\\left|\\right', '', latex_str)
    latex_str = re.sub(r'\s+', ' ', latex_str.strip())
    # 按照最新的建议，不使用 .lower()
    return latex_str

def debug_alignment():
    print("🧪 --- 启动 Hash 对齐性深度诊断 --- 🧪\n")
    
    # 1. 加载所有相关文件
    try:
        with open("data/processed/queries_full.json", 'r') as f:
            queries = json.load(f)
        with open("data/processed/formulas.json", 'r') as f:
            corpus = json.load(f)
        with open("data/processed/relevance_labels.json", 'r') as f:
            relevance = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ 缺少必要文件: {e}")
        return

    hash_gen = DualHashGenerator()
    found_case = False

    # 2. 遍历查询，寻找一个“本该匹配”的案例
    for topic_id, query_latex in queries.items():
        if topic_id not in relevance: continue
        
        # 获取该查询的所有标准答案 ID
        gt_ids = list(relevance[topic_id].keys())
        
        # 寻找库中存在的第一个标准答案
        for gt_id in gt_ids:
            gt_id_str = str(gt_id)
            if gt_id_str in corpus:
                found_case = True
                corpus_item = corpus[gt_id_str]
                corpus_latex = corpus_item['latex_norm']
                
                print(f"📍 [案例分析] Topic: {topic_id} -> Ground Truth ID: {gt_id_str}")
                print("-" * 60)
                
                # --- 核心对比：字符串级 ---
                print(f"【查询端 LaTeX (TSV)】:  {query_latex}")
                print(f"【语料端 LaTeX (JSON)】: {corpus_latex}")
                
                if query_latex == corpus_latex:
                    print("✅ 字符串完全对齐 (String Match: OK)")
                else:
                    print("❌ 字符串不一致！(String Match: FAILED)")
                    # 寻找第一个不一致的字符
                    for i, (c1, c2) in enumerate(zip(query_latex, corpus_latex)):
                        if c1 != c2:
                            print(f"   💡 差异点出现在第 {i} 位: '{c1}' vs '{c2}'")
                            break
                
                # --- 核心对比：哈希级 ---
                q_hash = hash_gen.generate_latex_hash(query_latex)
                c_hash = hash_gen.generate_latex_hash(corpus_latex)
                
                print(f"\n【查询端 Hash】: {q_hash}")
                print(f"【语料端 Hash】: {c_hash}")
                
                if q_hash == c_hash:
                    print("✅ 哈希生成一致 (Hash Match: OK)")
                else:
                    print("❌ 哈希不匹配！这说明 DualHashGenerator 内部存在不稳定逻辑。")

                print("-" * 60)
                # 每个查询只看第一个命中的 GT，或者只看前几个案例
                break 
        
        if found_case: break # 诊断出一个典型案例即可

    if not found_case:
        print("⚠️ 警告：在当前 formulas.json 中未找到任何标注的标准答案 ID。")
        print("💡 建议：请确认 prepare_final_arqmath.py 是否处理了包含标准答案的那些分片。")

if __name__ == "__main__":
    debug_alignment()