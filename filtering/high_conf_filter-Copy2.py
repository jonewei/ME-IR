# filtering/high_conf_filter.py

class HighConfidenceFilter:
    def __init__(self, sts_model):
        self.sts_model = sts_model
    
    def apply(self, query, candidates):
        """
        Filter candidates by STS score.
        """
        if not candidates:
            return []
        
        query_latex = query.get("latex") if isinstance(query, dict) else query
        
        # ✅ 启用真实的 STS 过滤
        filtered = []
        for cand in candidates:
            score = self.sts_model.predict_prob(query_latex, cand["latex"])
            if score >= self.sts_model.threshold:
                cand["sts_score"] = score
                filtered.append(cand)
        
        return filtered
