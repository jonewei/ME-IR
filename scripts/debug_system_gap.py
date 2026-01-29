import sqlite3
import json
import os
from pathlib import Path

def audit_data_consistency():
    db_path = "artifacts/formula_index.db"
    print(f"🔍 --- 正在审计数据库: {db_path} ---")
    
    if not os.path.exists(db_path):
        print(f"❌ 错误: 数据库文件不存在!")
        return

    with open("data/processed/queries_full.json", 'r') as f:
        queries = json.load(f)
    with open("data/processed/relevance_labels.json", 'r') as f:
        relevance = json.load(f)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查数据库总行数
    cursor.execute("SELECT COUNT(*) FROM formula_index")
    total_rows = cursor.fetchone()[0]
    print(f"📊 数据库总行数: {total_rows:,}")

    if total_rows == 0:
        print("❌ 严重警告: 数据库表为空！请检查 prepare_final_arqmath.py 中的 indexer.save_batch 调用。")
        return

    # --- 环节 A: ID 审计 ---
    print("\n🕵️ [审计 A]: ID 对齐检查")
    
    # 获取数据库里的一个 ID 样本
    cursor.execute("SELECT formula_id, h_dna FROM formula_index LIMIT 1")
    db_res = cursor.fetchone()
    db_sample_id = db_res[0]
    db_h_dna = db_res[1]

    # 获取标注里的一个 ID 样本
    sample_topic = list(relevance.keys())[0]
    qrel_sample_id = list(relevance[sample_topic].keys())[0]

    print(f"| 来源      | 值 (Value)          | 类型 (Type)        | 长度 (Len) |")
    print(f"|-----------|--------------------|-------------------|------------|")
    print(f"| Qrel (标注)| {repr(qrel_sample_id):<18} | {str(type(qrel_sample_id)):<17} | {len(str(qrel_sample_id)):<10} |")
    print(f"| DB (索引) | {repr(db_sample_id):<18} | {str(type(db_sample_id)):<17} | {len(str(db_sample_id)):<10} |")

    # --- 环节 B: DNA 冲突审计 ---
    print("\n🕵️ [审计 B]: DNA 骨架检查")
    sample_qid = list(queries.keys())[0]
    q_dna = queries[sample_qid].get('mathml_skel', "")
    print(f"查询 DNA 样本: {repr(q_dna[:50])}...")
    
    if any(c.isalpha() and c not in 'v' for c in q_dna):
        print("⚠️ 警告: 查询 DNA 包含未抽象化的变量！")

    conn.close()

if __name__ == "__main__":
    audit_data_consistency()