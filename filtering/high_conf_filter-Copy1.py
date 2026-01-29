
# # 高置信语义过滤，论文里的“非等价裁决”
# class HighConfidenceFilter:
#     """
#     Filter candidates using a high-confidence STS model.
#     """

#     def __init__(self, sts_model):
#         self.sts_model = sts_model

#     def apply(self, query: dict, candidates: list):
#         """
#         query: {"latex": str}
#         candidates: list of ranked formula dicts
#         """
#         filtered = []
#         for c in candidates:
#             if self.sts_model.is_similar(
#                 query["latex"], c["latex"]
#             ):
#                 filtered.append(c)
#         return filtered


class HighConfidenceFilter:
    """
    High-confidence semantic filter using STS.
    """

    def __init__(self, sts_model, threshold=0.5):
        self.sts = sts_model
        self.threshold = threshold

    def apply(self, query: dict, candidates: list) -> list:
        """
        Filter candidates based on semantic similarity score.
        """
        # ✅ 临时：禁用过滤，直接返回所有候选
        print(f"⚠️  [HighConfidenceFilter] TEMP: Filtering disabled, returning all {len(candidates)} candidates")
        return candidates
        
        # # 原始逻辑（暂时注释）
        # query_text = query.get("latex", "")
        # results = []
        # for c in candidates:
        #     score = self.sts.predict(query_text, c["latex"])
        #     if score >= self.threshold:
        #         c["sts_score"] = score
        #         results.append(c)
        # return results
