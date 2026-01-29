"""
🔧 修复版向量索引构建脚本 - 无检查点版本
适用于磁盘空间有限的情况，直接构建最终索引，不保存中间检查点
"""

import os
import json
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

# ==================== 配置参数 ====================
MODEL_NAME = 'math-similarity/Bert-MLM_arXiv-MP-class_zbMath'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "vector_index_full_v3.faiss"
MAPPING_PATH = ARTIFACTS_DIR / "vector_id_mapping_v3.json"

def build_index():
    """构建向量索引（无检查点，节省磁盘空间）"""
    
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    # 1. 加载模型
    print(f"🤖 正在加载模型: {MODEL_NAME}")
    print(f"   设备: {DEVICE}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    
    # 2. 读取元数据
    formulas_path = "data/processed/formulas.json"
    print(f"\n📖 正在读取 {formulas_path}...")
    
    with open(formulas_path, 'r', encoding='utf-8') as f:
        corpus_dict = json.load(f)
    
    fids = list(corpus_dict.keys())
    formulas = [corpus_dict[fid]['latex_norm'] for fid in fids]
    
    print(f"✅ 加载完成，总公式数: {len(fids):,}")
    
    # 数据质量检查
    print(f"\n📊 数据质量检查:")
    print(f"   前5个公式示例:")
    for i in range(min(5, len(formulas))):
        print(f"   [{fids[i]}]: {formulas[i][:80]}...")
    
    has_dollar = sum(1 for f in formulas[:1000] if '$' in f)
    if has_dollar > 0:
        print(f"   ⚠️  警告: 前1000条中有 {has_dollar} 条包含'$'符号")
        print(f"   ⚠️  请确认已使用修复版的prepare脚本重新生成formulas.json！")
    else:
        print(f"   ✅ 清洗检查通过：无$符号残留")

    # 3. 初始化 Faiss 索引
    dim = 768
    print(f"\n🔨 初始化 Faiss 索引 (IndexFlatIP, dim={dim})")
    index = faiss.IndexFlatIP(dim)
    
    # 4. 预估内存和磁盘需求
    estimated_memory_gb = len(formulas) * dim * 4 / (1024**3)
    print(f"\n💾 资源需求预估:")
    print(f"   - 内存占用: {estimated_memory_gb:.2f} GB")
    print(f"   - 最终索引文件: {estimated_memory_gb:.2f} GB")
    print(f"   - 映射文件: <1 MB")
    
    # 5. 批量编码与添加（无检查点）
    print(f"\n🚀 开始向量化 (Batch Size: {BATCH_SIZE})...")
    print(f"   ⚠️  磁盘空间有限，不保存检查点")
    print(f"   ⚠️  如果中断，需要重新运行整个脚本")
    
    chunk_size = 50000  # 每次处理5万条
    total_added = 0
    
    for i in tqdm(range(0, len(formulas), chunk_size), desc="总进度"):
        chunk_formulas = formulas[i : i + chunk_size]
        
        # 编码
        embeddings = model.encode(
            chunk_formulas,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # 添加到索引
        index.add(embeddings.astype('float32'))
        total_added += len(embeddings)
        
        # 每处理50万条显示一次进度
        if total_added % 500000 == 0:
            print(f"\n   ✅ 已处理 {total_added:,} / {len(formulas):,} ({total_added/len(formulas)*100:.1f}%)")

    # 6. 最终保存
    print(f"\n💾 正在保存最终索引...")
    print(f"   这可能需要几分钟，请耐心等待...")
    
    try:
        faiss.write_index(index, str(INDEX_PATH))
        print(f"   ✅ 索引保存成功: {INDEX_PATH}")
    except Exception as e:
        print(f"   ❌ 索引保存失败: {e}")
        print(f"   💡 可能原因: 磁盘空间不足")
        print(f"   💡 需要至少 {estimated_memory_gb:.1f} GB 可用空间")
        return False
    
    with open(MAPPING_PATH, 'w') as f:
        json.dump(fids, f)
    
    print(f"   ✅ 映射保存成功: {MAPPING_PATH}")
    
    # 7. 验证
    print(f"\n✅ 索引构建完成！")
    print(f"   索引文件: {INDEX_PATH}")
    print(f"   映射文件: {MAPPING_PATH}")
    print(f"   总向量数: {index.ntotal:,}")
    print(f"   索引类型: {type(index).__name__}")
    print(f"   维度: {index.d}")
    
    # 8. 快速测试
    print(f"\n🧪 快速测试...")
    test_query = formulas[0]
    test_emb = model.encode([test_query], normalize_embeddings=True, convert_to_numpy=True)
    D, I = index.search(test_emb.astype('float32'), 5)
    
    print(f"   查询: {test_query[:50]}...")
    print(f"   Top-1 距离: {D[0][0]:.4f} (应该接近1.0)")
    print(f"   Top-1 ID: {fids[I[0][0]]} (应该是自己)")
    
    if D[0][0] > 0.99:
        print(f"   ✅ 索引验证通过！")
        return True
    else:
        print(f"   ⚠️  警告: Top-1相似度异常，请检查normalize设置")
        return False

if __name__ == "__main__":
    success = build_index()
    if not success:
        print("\n❌ 构建失败！请检查错误信息")
        exit(1)
    else:
        print("\n🎉 构建成功！可以继续运行评测脚本")