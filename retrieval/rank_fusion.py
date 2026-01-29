from collections import defaultdict

def reciprocal_rank_fusion(vector_results, substructure_results, k=60, top_n=100):
    """
    RRF 融合算法实现
    :param vector_results: [(fid, score), ...] - 语义检索结果
    :param substructure_results: [(fid, score), ...] - 结构匹配结果
    :param k: RRF 常数，默认 60 效果最稳健
    :param top_n: 最终融合后返回的数量
    """
    rrf_scores = defaultdict(float)

    # 1. 累加向量检索排名贡献 (1 / (k + rank))
    for rank, item in enumerate(vector_results, start=1):
        # 处理 item 可能是元组或单纯 ID 的情况
        fid = str(item[0]) if isinstance(item, (tuple, list)) else str(item)
        rrf_scores[fid] += 1.0 / (k + rank)

    # 2. 累加子结构匹配排名贡献
    for rank, item in enumerate(substructure_results, start=1):
        fid = str(item[0]) if isinstance(item, (tuple, list)) else str(item)
        rrf_scores[fid] += 1.0 / (k + rank)

    # 3. 按最终 RRF 得分降序排序
    fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return fused_results[:top_n]