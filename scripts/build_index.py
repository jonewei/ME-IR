"""
Step 2: Build Approach0 structural index (Dual-Hash Version)

This script:
1. Loads combined formulas (Latex + MathML) from JSON
2. Filters out short noise formulas
3. Builds a dual-hash index (Latex-based and MathML-based)
4. Saves the index to disk
"""

import json
import logging
from pathlib import Path
import sys

# 确保能导入项目中的模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# from retrieval.approach0_hash import Approach0HashIndex
from retrieval.indexer import FormulaIndexer
from retrieval.approach0_hash import DualHashGenerator

# ========== Logging Setup ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_formulas(path):
    """
    加载由 prepare_combined_v2.py 生成的联合公式库
    """
    logger.info(f"📂 Loading combined formulas from {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        # 这里的 raw_formulas 是一个 Dict[visual_id, {formula_id, latex, mathml_skel}]
        formulas = json.load(f)
    
    logger.info(f"✅ Loaded {len(formulas):,} formulas")
    return formulas

def build_index(formulas):
    """
    构建双重哈希索引
    """
    logger.info("🔧 Building dual-hash index (Latex + MathML)...")
    
    index = Approach0HashIndex()
    
    indexed_count = 0
    skipped_count = 0
    
    # 遍历公式字典
    for i, (fid, formula_data) in enumerate(formulas.items()):
        latex = formula_data.get('latex', '')
        
        # ✅ 优化：过滤掉长度小于 3 的噪声公式（如单个算子、单个变量）
        # 这能显著减小索引体积并提升搜索质量
        if not latex or len(str(latex)) < 3:
            skipped_count += 1
            continue
            
        # ✅ 关键操作：将完整的 formula_data 传递给 add 方法
        # 内部逻辑会自动处理 formula_data["mathml_skel"] 用于生成高精度哈希
        index.add(formula_data)
        indexed_count += 1
        
        # 进度显示
        if (indexed_count) % 100000 == 0:
            logger.info(f"   Processed {i+1}/{len(formulas)} formulas...")
    
    logger.info(f"✅ Indexing complete!")
    logger.info(f"   - Total indexed: {indexed_count:,}")
    logger.info(f"   - Total skipped (noise): {skipped_count:,}")
    return index

def main():
    logger.info("=" * 60)
    logger.info("Step 2: Building Dual-Hash Structural Index")
    logger.info("=" * 60)
    
    # 输入路径
    input_path = "data/processed/formulas.json"
    if not Path(input_path).exists():
        logger.error(f"❌ Input file not found: {input_path}")
        logger.error("   Please run: python scripts/prepare_combined_v2.py first.")
        return
    
    # 1. 加载公式数据
    formulas = load_formulas(input_path)
    
    # 2. 构建索引
    index = build_index(formulas)
    
    # 3. 保存索引文件
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "approach0_index.pkl"
    
    logger.info(f"💾 Saving dual-hash index to {output_path}")
    index.save(str(output_path))
    
    # 4. 生成统计报告
    try:
        report = index.collision_report()
        logger.info("\n" + "=" * 30)
        logger.info("📊 Index Statistics")
        logger.info("-" * 30)
        logger.info(f"Total buckets: {report['total_buckets']:,}")
        logger.info(f"Avg bucket size: {report['avg_bucket_size']:.2f}")
        logger.info(f"Collision rate: {report['collision_rate']:.2%}")
        logger.info("=" * 30)
    except AttributeError:
        # 如果 Approach0HashIndex 中没写这个方法，则跳过
        pass

    logger.info("\n✅ Index building complete! Next: python scripts/run_eval.py")

if __name__ == "__main__":
    main()
