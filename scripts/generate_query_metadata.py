import json
import os

# 配置路径
INPUT_QUERIES = "data/processed/queries_full.json"
OUTPUT_METADATA = "data/query_metadata.json"

def generate_metadata():
    print(f"[*] 正在从 {INPUT_QUERIES} 提取元数据...")
    
    if not os.path.exists(INPUT_QUERIES):
        print(f"❌ 错误: 找不到输入文件 {INPUT_QUERIES}")
        return

    with open(INPUT_QUERIES, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    metadata = {}
    
    # 遍历所有查询进行统计
    # 假设 queries 格式为 { "301": "\frac{a}{b}", "302": "..." }
    for qid, latex in queries.items():
        # 统计指标 1: 字符长度 (Character Length)
        char_len = len(latex)
        
        # 统计指标 2: Token 数量 (按空格切分的粗略统计)
        token_count = len(latex.split())
        
        metadata[qid] = {
            "latex": latex,
            "length": char_len,     # run_experiments.py 默认读取这个字段
            "tokens": token_count
        }

    with open(OUTPUT_METADATA, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"✅ 成功生成 {OUTPUT_METADATA}，共计 {len(metadata)} 条查询元数据。")

if __name__ == "__main__":
    generate_metadata()