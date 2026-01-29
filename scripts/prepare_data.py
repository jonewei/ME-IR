import json
import os
from collections import defaultdict

# ================= 路径配置 =================
# 官方 Qrel 路径 (Task 2 为公式检索)
RAW_QREL_PATH = "/root/lanyun-tmp/math-expression-retrieval/data/arqmath3/qrel_task2_2022_official.tsv"

# 你的检索结果路径
RAW_STR_PATH = "results/ipi_results.txt" 
RAW_SEM_PATH = "results/semantic_results.txt"

# 结果保存路径
OUTPUT_QREL_JSON = "data/qrel_76_expert.json"
# ===============================================================

def generate_all():
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # --- 1. 转换官方 TSV 真值 ---
    print(f"[*] 正在读取官方真值: {RAW_QREL_PATH}")
    qrel_dict = defaultdict(dict)
    
    if os.path.exists(RAW_QREL_PATH):
        line_count = 0
        with open(RAW_QREL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                # 官方格式: Topic_ID \t Iteration(通常是0) \t Visual_ID \t Relevance
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    # 【核心修改点】：取 parts[2] 而非 parts[1]
                    topic_id = parts[0]
                    formula_id = parts[2] 
                    rel = parts[3]
                    
                    # 存入字典，确保 ID 为字符串以便匹配
                    qrel_dict[str(topic_id)][str(formula_id)] = int(rel)
                    line_count += 1
        
        with open(OUTPUT_QREL_JSON, 'w') as f:
            json.dump(qrel_dict, f)
        
        # 验证信息
        total_rel_docs = sum(len(v) for v in qrel_dict.values())
        print(f"✅ 成功生成: {OUTPUT_QREL_JSON}")
        print(f"   - 总查询数: {len(qrel_dict)}")
        print(f"   - 总相关记录数: {total_rel_docs}")
        if qrel_dict:
            first_qid = list(qrel_dict.keys())[0]
            print(f"   - 示例匹配: Topic {first_qid} -> IDs: {list(qrel_dict[first_qid].keys())[:3]}")
    else:
        print(f"❌ 错误: 找不到文件 {RAW_QREL_PATH}")

    # --- 2. 转换检索结果 (TREC to JSON) ---
    def convert_run(input_path, output_path):
        run_dict = defaultdict(dict)
        if os.path.exists(input_path):
            with open(input_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # TREC 格式: qid Q0 docid rank score tag
                        qid, _, docid, _, score = parts[:5]
                        # 确保 key 全是字符串
                        run_dict[str(qid)][str(docid)] = float(score)
            
            with open(output_path, 'w') as f:
                json.dump(run_dict, f)
            print(f"✅ 已生成 {output_path} (查询数: {len(run_dict)})")
        else:
            print(f"⚠️ 警告: 找不到检索结果文件 {input_path}")

    convert_run(RAW_STR_PATH, 'results/raw_str_scores.json')
    convert_run(RAW_SEM_PATH, 'results/raw_sem_scores.json')

if __name__ == "__main__":
    generate_all()