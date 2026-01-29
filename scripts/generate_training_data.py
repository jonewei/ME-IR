# scripts/generate_training_data.py

"""
Generate training data for ranking model from relevance labels.
"""

import json
import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.approach0_hash import Approach0HashIndex
from retrieval.recall_api import StructuralRecall
from ranking import MathBERTEmbedder, FeatureBuilder

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_training_pairs(queries, labels, formulas, index, embedder, feature_builder):
    """
    从 relevance labels 生成训练数据
    
    Returns:
        train_data: List of feature vectors
        train_labels: List of relevance labels (0-3)
        train_groups: List of group sizes (queries)
    """
    logger.info("🔧 Generating training data from relevance labels...")
    
    recall = StructuralRecall(index)
    
    train_data = []
    train_labels = []
    train_groups = []
    
    for i, (qid, query_latex) in enumerate(queries.items()):
        if qid not in labels:
            continue
        
        logger.info(f"Processing query {i+1}/{len(queries)}: {qid}")
        
        # 1. 召回候选
        query_dict = {"query_id": qid, "latex": query_latex}
        candidates = recall.retrieve(query_dict, topk=100)
        
        if not candidates:
            logger.warning(f"  No candidates for query {qid}")
            continue
        
        # 2. 获取查询嵌入
        query_emb = embedder.encode(query_latex)
        
        # 3. 为每个候选生成特征
        query_features = []
        query_labels = []
        
        for cand in candidates:
            # 获取候选嵌入
            cand_emb = embedder.encode(cand["latex"])
            
            # 构建特征
            features = feature_builder.build(
                query_emb=query_emb,
                cand_emb=cand_emb,
                cand=cand,
                query_dict=query_dict
            )
            
            # 获取标签（相关性）
            cand_id = cand["formula_id"]
            relevance = labels[qid].get(cand_id, 0)  # 0 = 不相关
            
            query_features.append(features)
            query_labels.append(relevance)
        
        # 添加到训练集
        train_data.extend(query_features)
        train_labels.extend(query_labels)
        train_groups.append(len(query_features))
        
        logger.info(f"  Added {len(query_features)} pairs, "
                   f"relevant: {sum(1 for l in query_labels if l > 0)}")
    
    logger.info(f"\n✅ Training data generated:")
    logger.info(f"   Total pairs: {len(train_data)}")
    logger.info(f"   Queries: {len(train_groups)}")
    logger.info(f"   Relevant pairs: {sum(1 for l in train_labels if l > 0)}")
    
    return np.array(train_data), np.array(train_labels), train_groups


def main():
    logger.info("=" * 60)
    logger.info("Generating Training Data")
    logger.info("=" * 60)
    
    # 1. 加载所有必需数据
    logger.info("\n📂 Loading data...")
    
    with open("data/processed/formulas.json") as f:
        formulas = json.load(f)
    
    with open("data/processed/queries.json") as f:
        queries = json.load(f)
    
    with open("data/processed/relevance_labels.json") as f:
        labels = json.load(f)
    
    # 标准化 labels 格式
    for qid in labels:
        if isinstance(labels[qid], list):
            labels[qid] = {fid: 1 for fid in labels[qid]}
    
    index = Approach0HashIndex()
    index.load("artifacts/approach0_index.pkl")
    
    logger.info(f"  Formulas: {len(formulas)}")
    logger.info(f"  Queries: {len(queries)}")
    logger.info(f"  Index: {len(index.index)} buckets")
    
    # 2. 初始化模型
    logger.info("\n🔧 Initializing models...")
    embedder = MathBERTEmbedder()
    feature_builder = FeatureBuilder(num_features=20)
    
    # 3. 生成训练数据
    train_data, train_labels, train_groups = generate_training_pairs(
        queries, labels, formulas, index, embedder, feature_builder
    )
    
    # 4. 保存
    output_dir = Path("data/ranking")
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / "train_data.npy", train_data)
    np.save(output_dir / "train_labels.npy", train_labels)
    
    with open(output_dir / "train_groups.json", 'w') as f:
        json.dump(train_groups, f)
    
    logger.info(f"\n💾 Training data saved to {output_dir}/")
    logger.info("   - train_data.npy")
    logger.info("   - train_labels.npy")
    logger.info("   - train_groups.json")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Training data generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
