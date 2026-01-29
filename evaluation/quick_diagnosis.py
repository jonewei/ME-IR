"""
🔧 修复版快速诊断脚本
解决了 formulas.json 数据结构解析问题
"""

import json
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch

# ==================== 配置 ====================
MODEL_NAME = 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_PATH = "artifacts/vector_index_full_v3.faiss"
MAPPING_PATH = "artifacts/vector_id_mapping_v3.json"
FORMULAS_PATH = "data/processed/formulas.json"
QUERY_PATH = "data/processed/queries_full.json"

def clean_latex(latex_str):
    if not latex_str: 
        return ""
    latex_str = re.sub(r'\$\$?|\\\[|\\\]', '', latex_str)
    latex_str = re.sub(r'\\dfrac|\\tfrac', r'\\frac', latex_str)
    latex_str = re.sub(r'\\left|\\right', '', latex_str)
    latex_str = re.sub(r'\s+', ' ', latex_str.strip())
    return latex_str.lower()

def run_diagnosis():
    print("="*70)
    print("🔬 开始快速诊断...")
    print("="*70)
    
    # ==================== 检查1: 文件存在性 ====================
    print("\n[检查1] 文件完整性检查...")
    files_to_check = {
        'Faiss索引': INDEX_PATH,
        'ID映射': MAPPING_PATH,
        '公式元数据': FORMULAS_PATH,
        '查询数据': QUERY_PATH
    }
    
    all_files_exist = True
    for name, path in files_to_check.items():
        exists = Path(path).exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {name}: {path}")
        if not exists:
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ 文件不完整！请先运行prepare和build脚本。")
        return
    
    # ==================== 检查2: 数据结构检查 ====================
    print("\n[检查2] 数据结构检查...")
    
    # 🔧 修复：正确读取formulas.json（完整JSON）
    print("   正在读取 formulas.json...")
    with open(FORMULAS_PATH, 'r') as f:
        formulas_dict = json.load(f)
    
    # 检查数据结构
    sample_ids = list(formulas_dict.keys())[:3]
    print(f"   ✅ formulas.json 加载成功，共 {len(formulas_dict):,} 条公式")
    print(f"   前3个ID: {sample_ids}")
    
    # 检查第一个条目的结构
    first_id = sample_ids[0]
    first_item = formulas_dict[first_id]
    
    print(f"\n   示例条目 [{first_id}]:")
    if isinstance(first_item, dict):
        print(f"      类型: 字典 ✅")
        print(f"      字段: {list(first_item.keys())}")
        
        # 检查latex_norm
        if 'latex_norm' in first_item:
            latex_norm = first_item['latex_norm']
            print(f"      latex_norm: {latex_norm[:80]}...")
            
            if '$' in latex_norm:
                print(f"   ⚠️  警告: latex_norm包含$符号！")
            else:
                print(f"   ✅ latex_norm已正确清洗（无$符号）")
        else:
            print(f"   ❌ 缺少 latex_norm 字段！")
    else:
        print(f"   ❌ 错误: 期望字典，实际是 {type(first_item)}")
    
    # 检查queries.json
    print("\n   正在读取 queries.json...")
    with open(QUERY_PATH, 'r') as f:
        queries_dict = json.load(f)
    
    sample_qids = list(queries_dict.keys())[:3]
    print(f"   ✅ queries.json 加载成功，共 {len(queries_dict)} 条查询")
    print(f"   前3个查询ID: {sample_qids}")
    
    first_qid = sample_qids[0]
    first_query = queries_dict[first_qid]
    
    print(f"\n   示例查询 [{first_qid}]:")
    print(f"      类型: {type(first_query)}")
    
    if isinstance(first_query, dict):
        print(f"      字段: {list(first_query.keys())}")
        query_text = first_query.get('latex') or first_query.get('latex_norm', '')
    elif isinstance(first_query, str):
        query_text = first_query
    else:
        print(f"   ⚠️  未知的查询格式")
        query_text = ""
    
    if query_text:
        print(f"      查询内容: {query_text[:80]}...")
        if '$' in query_text:
            print(f"   ⚠️  查询包含$符号")
    
    # ==================== 检查3: 模型和索引加载 ====================
    print("\n[检查3] 模型和索引加载检查...")
    
    try:
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        print(f"   ✅ 模型加载成功: {MODEL_NAME}")
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        return
    
    try:
        index = faiss.read_index(INDEX_PATH)
        print(f"   ✅ Faiss索引加载成功")
        print(f"      - 向量数量: {index.ntotal:,}")
        print(f"      - 维度: {index.d}")
        print(f"      - 索引类型: {type(index).__name__}")
    except Exception as e:
        print(f"   ❌ 索引加载失败: {e}")
        return
    
    with open(MAPPING_PATH, 'r') as f:
        fids = json.load(f)
    print(f"   ✅ ID映射加载成功: {len(fids):,} 条")
    
    # ==================== 检查4: 向量一致性测试 ====================
    print("\n[检查4] 向量编码一致性测试...")
    
    # 测试公式：从索引中取第一个
    test_id = fids[0]
    
    # 从formulas_dict获取对应的latex
    if test_id not in formulas_dict:
        print(f"   ❌ 错误: ID {test_id} 在formulas.json中不存在")
        return
    
    test_item = formulas_dict[test_id]
    if isinstance(test_item, dict) and 'latex_norm' in test_item:
        test_latex_stored = test_item['latex_norm']
    elif isinstance(test_item, dict) and 'latex' in test_item:
        test_latex_stored = test_item['latex']
    else:
        print(f"   ❌ 无法从条目中提取latex")
        return
    
    print(f"\n   测试公式ID: {test_id}")
    print(f"   存储的latex: {test_latex_stored[:80]}...")
    
    # 模拟查询端处理
    test_latex_clean = clean_latex(test_latex_stored)
    print(f"   clean后: {test_latex_clean[:80]}...")
    
    if test_latex_stored != test_latex_clean:
        print(f"   ⚠️  警告: 存储值与clean值不一致")
        print(f"      这可能导致向量不匹配！")
    else:
        print(f"   ✅ 存储值与clean值一致")
    
    # 编码查询
    query_emb = model.encode(
        [test_latex_clean], 
        normalize_embeddings=True, 
        convert_to_numpy=True
    ).astype('float32')
    
    # 搜索
    D, I = index.search(query_emb, 5)
    
    print(f"\n   Top-5 检索结果:")
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        result_id = fids[idx]
        is_self = "⭐ [自己]" if idx == 0 else ""  # 第一个ID应该是自己
        print(f"      {rank+1}. 索引位置={idx}, ID={result_id}, 距离={dist:.4f} {is_self}")
    
    # 关键检查
    if I[0][0] == 0 and D[0][0] > 0.99:
        print(f"\n   ✅ 向量编码一致性测试通过！")
    else:
        print(f"\n   ❌ 严重问题: 公式无法检索到自己！")
        print(f"      - 期望索引位置: 0")
        print(f"      - 实际索引位置: {I[0][0]}")
        print(f"      - Top-1相似度: {D[0][0]:.4f} (应该>0.99)")
        print(f"\n   可能原因:")
        print(f"      1. 索引端和查询端的clean_latex不一致")
        print(f"      2. normalize_embeddings设置不一致")
        print(f"      3. formulas.json与索引不对应")
    
    # ==================== 检查5: 真实查询测试 ====================
    print("\n[检查5] 真实查询测试...")
    
    # 测试前3个查询
    for qid in sample_qids[:3]:
        query_raw = queries_dict[qid]
        
        if isinstance(query_raw, dict):
            query_text = query_raw.get('latex') or query_raw.get('latex_norm', '')
        else:
            query_text = query_raw
        
        query_clean = clean_latex(query_text)
        
        print(f"\n   查询 [{qid}]:")
        print(f"      原始: {query_text[:60]}...")
        print(f"      clean: {query_clean[:60]}...")
        
        query_emb = model.encode([query_clean], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
        D, I = index.search(query_emb, 3)
        
        print(f"      Top-3结果:")
        for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
            result_id = fids[idx]
            result_item = formulas_dict.get(result_id, {})
            if isinstance(result_item, dict):
                result_latex = result_item.get('latex_norm', 'N/A')[:40]
            else:
                result_latex = 'N/A'
            print(f"         {rank+1}. 距离={dist:.4f}, latex={result_latex}...")
    
    # ==================== 总结 ====================
    print("\n" + "="*70)
    print("📊 诊断总结")
    print("="*70)
    print("✅ 已完成所有检查")
    print("\n建议:")
    print("  1. 如果向量一致性测试失败 → 检查prepare脚本的clean_latex")
    print("  2. 如果Recall仍然很低 → 运行详细的错误分析")
    print("  3. 运行 hash_recall_evaluator.py 测试Stage 1")
    print("="*70)

if __name__ == "__main__":
    run_diagnosis()