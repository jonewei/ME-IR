import logging
logger = logging.getLogger(__name__)

class HighConfidenceFilter:
    def __init__(self, sts_model):
        self.sts_model = sts_model
    
    def apply(self, query, candidates):
        query_latex = query["latex"]
        
        if not candidates:
            return []
            
        logger.info(f"🔍 Applying STS filter (threshold={self.sts_model.threshold}) to {len(candidates)} candidates")
        
        filtered = self.sts_model.apply_threshold(query_latex, candidates)
        
        filter_rate = 1 - len(filtered)/len(candidates) if candidates else 0
        logger.info(f"📊 STS filtered: {len(filtered)} remaining (过滤率: {filter_rate:.1%})")
        
        if not filtered:
            logger.warning("⚠️  STS过滤后无结果，回退到粗排结果")
            return candidates  # 回退，避免全空
            
        return filtered
