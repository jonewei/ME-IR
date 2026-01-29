# import json
# import torch
# import faiss
# import sys
# from transformers import AutoTokenizer, AutoModel
# from pathlib import Path
# from tqdm import tqdm

# def check_top_results():
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     MODEL_NAME = "witiko/mathberta"
    
#     print("🚀 正在初始化语义检查工具...")

#     # 1. 加载映射表 (这个比较小，很快)
#     mapping_path = Path("artifacts/vector_id_mapping_pq.json")
#     if not mapping_path.exists():
#         print(f"❌ 错误: 找不到映射文件 {mapping_path}")
#         return
#     with open(mapping_path, 'r') as f:
#         fids = json.load(f)
#     print(f"✅ 已加载映射表，包含 {len(fids):,} 条公式 ID")

#     # 2. 加载 Faiss 索引
#     index_path = Path("artifacts/vector_index_pq.faiss")
#     print(f"📦 正在加载 Faiss 索引 ({index_path.stat().st_size / 1024**2:.2f} MB)...")
#     index = faiss.read_index(str(index_path))
#     print("✅ 索引加载完成")

#     # 3. 加载 MathBERT 模型
#     print(f"🤖 正在加载模型 {MODEL_NAME} 到 {DEVICE}...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
#     model.eval()
#     print("✅ 模型准备就绪")

#     # 4. 优化加载 formulas.json (关键步骤)
#     print("📖 正在读取公式元数据 (仅读取前 100,000 条以节省内存)...")
#     corpus = {}
#     with open("data/processed/formulas.json", 'r', encoding='utf-8') as f:
#         # 使用流式思路模拟进度条（虽然 json.load 是阻塞的，但我们可以先读取前 100k 条对应的元数据）
#         # 如果你之前只索引了 10w 条，这里我们也只取前 10w 条
#         full_corpus = json.load(f)
#         for i, fid in enumerate(tqdm(fids, desc="映射元数据")):
#             if fid in full_corpus:
#                 corpus[fid] = full_corpus[fid]
#         del full_corpus # 立即释放全量大字典

#     # 5. 准备查询
#     with open("data/processed/queries_full.json", 'r') as f:
#         queries = json.load(f)
    
#     # 挑选第 50 个查询（避开最简单的，找个有挑战性的）
#     test_qid = list(queries.keys())[50] 
#     query_latex = queries[test_qid]['latex_norm']
    
#     print("-" * 50)
#     print(f"🔎 [查询主题]: {test_qid}")
#     print(f"🔎 [查询公式]: {query_latex}")
#     print("-" * 50)
    
#     # 6. 执行向量编码与检索
#     print("🧠 正在进行深度语义编码...")
#     inputs = tokenizer([query_latex], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         q_emb = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
#     faiss.normalize_L2(q_emb)
    
#     print("🔍 正在语义空间搜索 Top 5...")
#     D, I = index.search(q_emb, 5)
    
#     print("\n🎯 [语义相似度检索结果]:")
#     for i, idx in enumerate(I[0]):
#         if idx == -1: continue
#         fid = fids[idx]
#         score = D[0][i]
#         res_latex = corpus.get(fid, {}).get('latex_norm', "未知内容")
#         print(f"Rank {i+1} [相似度: {score:.4f}]:")
#         print(f"   ID: {fid}")
#         print(f"   LaTeX: {res_latex}")
#         print("-" * 20)

# if __name__ == "__main__":
#     check_top_results()
import json
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def get_latex_robust(target_ids, json_path):
    """
    最稳健的流式 ID 查找：逐行读取并解析 JSON 逻辑块
    """
    results = {}
    target_ids = {str(tid) for tid in target_ids}
    print(f"📖 正在全量扫描 1300 万条元数据，寻找 ID: {target_ids}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            current_id = None
            for line in f:
                # 匹配行如: "2742591": {
                line = line.strip()
                for fid in list(target_ids):
                    if line.startswith(f'"{fid}":'):
                        current_id = fid
                        break
                
                # 如果当前行在某个需要的 ID 块内，寻找 latex_norm
                if current_id and '"latex_norm":' in line:
                    # 提取引号内的内容
                    start = line.find('": "') + 4
                    end = line.rfind('"')
                    if start > 3 and end > start:
                        results[current_id] = line[start:end]
                        target_ids.remove(current_id)
                        current_id = None
                
                if not target_ids:
                    break
    except Exception as e:
        print(f"❌ 扫描过程中出错: {e}")
    return results

def check_semantic():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "witiko/mathberta"
    
    # 1. 加载映射与索引
    with open("artifacts/vector_id_mapping_pq.json", 'r') as f:
        fids = json.load(f)
    index = faiss.read_index("artifacts/vector_index_pq.faiss")
    
    # 2. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    # 3. 执行查询 (勾股定理)
    # query_latex = r"a^2 + b^2 = c^2"
    query_latex = r"x^2 + y^2 = z^2"
    print(f"\n🔎 查询公式: {query_latex}")

    inputs = tokenizer([query_latex], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        q_emb = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
    faiss.normalize_L2(q_emb)
    
    D, I = index.search(q_emb, 5)
    result_ids = [fids[idx] for idx in I[0] if idx != -1]

    # 4. 鲁棒提取内容
    content_map = get_latex_robust(result_ids, "data/processed/formulas.json")

    print("\n🎯 [语义相似度检索结果]:")
    for i, idx in enumerate(I[0]):
        fid = fids[idx]
        latex = content_map.get(str(fid), "内容提取失败")
        print(f"Rank {i+1} [相似度: {D[0][i]:.4f}]:")
        print(f"   ID: {fid}")
        print(f"   LaTeX: {latex}")
        print("-" * 20)

if __name__ == "__main__":
    check_semantic()