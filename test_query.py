import sqlite3
import json

def diagnostic_test():
    # 1. 加载数据
    with open("data/processed/queries_full.json", 'r') as f:
        queries = json.load(f)
    with open("data/processed/relevance_labels.json", 'r') as f:
        relevance = json.load(f)

    conn = sqlite3.connect("artifacts/formula_index.db")
    cursor = conn.cursor()

    print("🚀 正在寻找一个存在于数据库中的相关文档进行对比...")

    found = False
    for topic_id, rel_docs in relevance.items():
        # 寻找该主题对应的查询 ID
        qid = next((k for k in queries.keys() if topic_id in k), None)
        if not qid: continue

        for doc_id in rel_docs.keys():
            # 尝试查找数据库中是否存在该文档 (尝试原 ID 和带 v 前缀的 ID)
            cursor.execute("SELECT formula_id, h_dna FROM formula_index WHERE formula_id IN (?, ?)", (doc_id, f"v{doc_id}"))
            res = cursor.fetchone()
            
            if res:
                db_fid, db_h_dna = res
                print(f"\n✅ 找到匹配对！")
                print(f"主题: {topic_id} | 查询 ID: {qid}")
                print(f"文档 ID: {db_fid}")
                print("-" * 30)
                print(f"Query DNA 样例: {queries[qid]['mathml_skel'][:100]}")
                print(f"DB DNA 哈希值:  {db_h_dna}")
                print("-" * 30)
                print("💡 建议：现在我们知道了哈希不匹配。")
                found = True
                break
        if found: break
    
    if not found:
        print("❌ 警告：在当前 50 个分片中未找到任何 Qrel 标注的相关文档。")
    conn.close()

if __name__ == "__main__":
    diagnostic_test()