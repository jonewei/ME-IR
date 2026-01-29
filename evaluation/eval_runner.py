
"""
Enhanced evaluation runner with TREC qrel support and corrected nDCG calculation.

Key fixes:
1. ✅ Support for TSV qrel format (B.301 0 60069 3)
2. ✅ Corrected nDCG@K calculation (fixed IDCG truncation)
3. ✅ Robust error handling
4. ✅ Progress bar integration
"""

import numpy as np
from tqdm import tqdm
import json
import logging

logger = logging.getLogger(__name__)


def load_qrel_labels(qrel_path):
    """
    加载 TREC qrel 格式标签文件
    
    格式: query_id  iteration  doc_id  relevance
    示例: B.301     0          60069   3
    
    Returns:
        Dict[query_id, Dict[doc_id, relevance_score]]
    """
    labels = {}
    
    logger.info(f"📂 Loading qrel labels from {qrel_path}")
    
    with open(qrel_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            
            # 验证格式
            if len(parts) < 4:
                logger.warning(f"Line {line_num}: Invalid format (expected 4 fields, got {len(parts)})")
                continue
            
            qid, iteration, doc_id, rel = parts[:4]
            
            # 初始化查询字典
            if qid not in labels:
                labels[qid] = {}
            
            # 存储相关性分数
            try:
                labels[qid][doc_id] = int(rel)
            except ValueError:
                logger.warning(f"Line {line_num}: Invalid relevance score '{rel}', skipping")
                continue
    
    logger.info(f"✅ Loaded {len(labels)} queries with relevance labels")
    
    # 统计信息
    total_rels = sum(len(docs) for docs in labels.values())
    logger.info(f"   Total relevance judgments: {total_rels}")
    
    return labels


def calculate_metrics(all_results, labels):
    """
    计算评估指标：Recall@K, MAP, nDCG@K
    
    Args:
        all_results: Dict[query_id, List[candidate_dict]]
        labels: Dict[query_id, Dict[doc_id, relevance_score]]
    
    Returns:
        Dict with averaged metrics
    """
    maps = []
    ndcgs = []
    recalls = []
    
    # 统计未找到标签的查询
    missing_labels = 0
    
    for qid, candidates in all_results.items():
        if qid not in labels:
            logger.debug(f"Query {qid} has no ground truth labels, skipping")
            missing_labels += 1
            continue
        
        gt_dict = labels[qid]  # Dict[doc_id, relevance_score]
        
        # 提取检索到的 ID 列表
        pred_ids = [c['formula_id'] for c in candidates]
        
        # --- 1. Recall@K (二值化版本) ---
        hits = sum(1 for fid in pred_ids if fid in gt_dict)
        recalls.append(1 if hits > 0 else 0)
        
        # --- 2. Average Precision (AP) ---
        ap = 0.0
        relevant_found = 0
        
        for i, fid in enumerate(pred_ids):
            if fid in gt_dict and gt_dict[fid] > 0:  # 考虑相关性分数
                relevant_found += 1
                ap += relevant_found / (i + 1)
        
        total_relevant = sum(1 for score in gt_dict.values() if score > 0)
        maps.append(ap / max(1, total_relevant))
        
        # --- 3. nDCG@K ---
        # 计算 DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, fid in enumerate(pred_ids):
            rel = gt_dict.get(fid, 0)
            dcg += (2**rel - 1) / np.log2(i + 2)
        
        # ✅ 修正 IDCG 计算：使用固定的 K
        k = len(pred_ids)
        ideal_rels = sorted(gt_dict.values(), reverse=True)[:k]  # ← 修正：取 Top-K 个最相关的
        
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += (2**rel - 1) / np.log2(i + 2)
        
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    
    # 日志输出
    if missing_labels > 0:
        logger.warning(f"⚠️  {missing_labels} queries have no ground truth labels")
    
    logger.info(f"📊 Evaluated {len(recalls)} queries")
    
    return {
        "Recall@K": float(np.mean(recalls)) if recalls else 0.0,
        "MAP": float(np.mean(maps)) if maps else 0.0,
        "nDCG@K": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "num_evaluated_queries": len(recalls)
    }


def evaluate(pipeline, queries, labels):
    """
    运行评估循环并返回指标和全量结果
    
    Args:
        pipeline: SearchPipeline instance
        queries: List[dict] with query_id and latex
        labels: Dict[query_id, Dict[doc_id, relevance]]
    
    Returns:
        (metrics: dict, all_results: dict)
    """
    all_results = {}
    
    # 进度条初始化
    progress_bar = tqdm(queries, desc="🔎 Evaluating", unit="query", leave=True)
    
    failed_count = 0
    
    for query in progress_bar:
        qid = query["query_id"]
        progress_bar.set_postfix({"current_id": qid, "failed": failed_count})
        
        # 执行检索
        try:
            results = pipeline.search(query)
            all_results[qid] = results
        except Exception as e:
            logger.error(f"❌ Error processing query {qid}: {e}")
            all_results[qid] = []
            failed_count += 1
    
    # 统计失败率
    if failed_count > 0:
        logger.warning(f"⚠️  {failed_count}/{len(queries)} queries failed")
    
    # 计算最终指标
    metrics = calculate_metrics(all_results, labels)
    
    return metrics, all_results


def save_trec_run(results, output_path, run_id="math-retrieval-system"):
    """
    保存为官方 TREC 评测格式
    
    格式: query_id Q0 doc_id rank score run_id
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for qid, candidates in results.items():
                for rank, cand in enumerate(candidates, 1):
                    # 尝试多个分数字段
                    score = cand.get('final_score', cand.get('rank_score', 0.0))
                    f.write(f"{qid} Q0 {cand['formula_id']} {rank} {score:.6f} {run_id}\n")
        
        logger.info(f"💾 TREC run saved to {output_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save TREC run: {e}")



