import faiss
import torch
import json
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
from argparse import ArgumentParser
from tqdm import tqdm

def run_semantic_search(args):
    # 1. 环境准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    # 2. 加载 zbMath 专用模型与分词器
    print(f"[*] 正在加载预训练模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device)
    model.eval()

    # 3. 加载已经生成的 v4 FAISS 索引
    print(f"[*] 正在加载 FAISS 索引: {args.index_path}")
    if not os.path.exists(args.index_path):
        raise FileNotFoundError(f"找不到索引文件: {args.index_path}")
    index = faiss.read_index(args.index_path)
    
    # 4. 关键步骤：解析 JSON 映射文件
    print(f"[*] 正在解析 JSON 映射表: {args.json_map_path}")
    with open(args.json_map_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    # 将 JSON 转换为列表，确保 Row ID (0, 1, 2...) 能精准对应 Formula ID
    if isinstance(mapping_data, list):
        # 格式 A: ["id1", "id2", ...]
        id_map = mapping_data
    elif isinstance(mapping_data, dict):
        # 格式 B: {"0": "id1", "1": "id2", ...}
        # 按照索引顺序重建列表，防止字典无序或 Key 错位
        print("[*] 检测到字典格式映射，正在按索引排序...")
        max_idx = len(mapping_data)
        id_map = [mapping_data.get(str(i)) or mapping_data.get(i) for i in range(max_idx)]
    else:
        raise ValueError("不支持的 JSON 映射格式，必须为 List 或 Dict。")

    if len(id_map) != index.ntotal:
        print(f"⚠️ 警告: 映射表条目数({len(id_map)})与索引向量总数({index.ntotal})不一致！")

    # 5. 读取查询文件 (JSON 格式: {qid: latex})
    print(f"[*] 正在读取查询任务: {args.query_path}")
    with open(args.query_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # 6. 执行检索循环
    print(f"[*] 开始语义检索，目标语料库规模: {index.ntotal}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for qid, latex in tqdm(queries.items(), desc="Semantic Searching"):
            # A. 将查询 LaTeX 编码为向量
            with torch.no_grad():
                inputs = tokenizer(
                    latex, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=256 
                ).to(device)
                
                outputs = model(**inputs)
                # 提取 CLS 向量作为公式的全局特征表示
                q_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')
                # L2 归一化是确保内积检索等于余弦相似度的核心
                faiss.normalize_L2(q_emb)

            # B. 在 FAISS 中执行 Top-K 向量搜索
            scores, indices = index.search(q_emb, args.top_k)

            # C. 写入结果文件 (TREC 标准格式)
            # qid Q0 docid rank score tag
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx == -1 or idx >= len(id_map): 
                    continue
                formula_id = id_map[idx]
                f_out.write(f"{qid} Q0 {formula_id} {rank+1} {score:.6f} zbMath_Semantic\n")

    print(f"✅ 检索完成！结果已保存至: {args.output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="LS-MIR 语义流检索 (JSON 映射版)")
    
    # 路径配置
    parser.add_argument("--model_path", type=str, default="math-similarity/Bert-MLM_arXiv-MP-class_zbMath")
    parser.add_argument("--index_path", type=str, default="artifacts/vector_index_full_v4.faiss")
    parser.add_argument("--json_map_path", type=str, default="artifacts/vector_id_mapping_v4.json")
    
    # 输入与输出
    parser.add_argument("--query_path", type=str, default="data/processed/queries_full.json")
    parser.add_argument("--output_path", type=str, default="results/semantic_results.txt")  
    
    # 检索参数
    parser.add_argument("--top_k", type=int, default=1000, help="返回相似公式的数量")

    args = parser.parse_args()
    run_semantic_search(args)