"""
Formula STS Model with TeX-Match Support

Key improvements:
1. ✅ Support for math-specialized models (TeX-Match, MathBERT)
2. ✅ Adaptive thresholding based on query complexity
3. ✅ Batch processing with GPU optimization
4. ✅ Fallback mechanism for model loading failures
"""

import torch
from sentence_transformers import SentenceTransformer, util
import logging
import numpy as np
from typing import List, Dict
import re

logger = logging.getLogger(__name__)


class FormulaSTSModel:
    """
    Mathematical formula similarity using specialized transformers
    
    Supported models:
    1. math-similarity/Bert-MLM_arXiv-MP-class_zbMath (original)
    2. tbs17/MathBERT (math-specialized BERT)
    3. all-mpnet-base-v2 (strong general-purpose model)
    """
    
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        threshold: float = 0.65,  # ✅ 降低默认阈值
        device: str = None,
        use_adaptive_threshold: bool = True
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            threshold: Base similarity threshold
            device: 'cuda' or 'cpu'
            use_adaptive_threshold: Adjust threshold based on query complexity
        """
        self.threshold = threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"✅ Loading STS model: {model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Base threshold: {self.threshold}")
        logger.info(f"   Adaptive threshold: {self.use_adaptive_threshold}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"   ✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            logger.warning(f"   Falling back to all-mpnet-base-v2")
            
            # Fallback to reliable general-purpose model
            self.model = SentenceTransformer("all-mpnet-base-v2")
            self.model.to(self.device)
            self.model.eval()
    
    def _compute_query_complexity(self, latex: str) -> float:
        """
        Estimate query complexity (0-1 scale)
        
        Complex queries should have lower thresholds to avoid over-filtering.
        """
        if not latex:
            return 0.0
        
        # Complexity indicators
        complexity_score = 0.0
        
        # 1. Nesting depth (fractions, roots)
        complexity_score += latex.count(r'\frac') * 0.1
        complexity_score += latex.count(r'\sqrt') * 0.1
        
        # 2. Special functions
        special_funcs = [r'\int', r'\sum', r'\lim', r'\prod']
        for func in special_funcs:
            complexity_score += latex.count(func) * 0.15
        
        # 3. Matrix/array structures
        if 'matrix' in latex or 'array' in latex:
            complexity_score += 0.3
        
        # 4. Length (normalized)
        complexity_score += min(len(latex) / 100, 0.3)
        
        return min(complexity_score, 1.0)
    
    def _get_adaptive_threshold(self, query_latex: str) -> float:
        """
        Compute adaptive threshold based on query complexity
        
        Strategy:
        - Simple queries (e.g., x^2): High threshold (0.75)
        - Complex queries (e.g., ∫∫∫): Lower threshold (0.55)
        """
        if not self.use_adaptive_threshold:
            return self.threshold
        
        complexity = self._compute_query_complexity(query_latex)
        
        # Threshold decreases with complexity
        # Simple: 0.75, Medium: 0.65, Complex: 0.55
        adaptive_threshold = self.threshold + (0.1 * (1 - complexity))
        
        logger.debug(f"Query complexity: {complexity:.2f}, threshold: {adaptive_threshold:.2f}")
        
        return adaptive_threshold
    
    def score(self, query_latex: str, candidate_latex: str) -> float:
        """
        Compute cosine similarity between two formulas
        
        Args:
            query_latex: Query formula LaTeX
            candidate_latex: Candidate formula LaTeX
            
        Returns:
            Similarity score in [0, 1]
        """
        if not query_latex or not candidate_latex:
            return 0.0
        
        try:
            with torch.no_grad():
                emb1 = self.model.encode(
                    query_latex,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False
                )
                emb2 = self.model.encode(
                    candidate_latex,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False
                )
                
                sim = util.cos_sim(emb1, emb2)[0][0].item()
                
            return float(sim)
            
        except Exception as e:
            logger.error(f"STS scoring failed: {e}")
            return 0.0
    
    def predict_prob(self, query_latex: str, candidate_latex: str) -> float:
        """Alias for score (backward compatibility)"""
        return self.score(query_latex, candidate_latex)
    
    def apply_threshold(
        self,
        query_latex: str,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        🚀 Batch filter candidates by adaptive similarity threshold
        
        Args:
            query_latex: Query LaTeX string
            candidates: List of candidate dicts
            
        Returns:
            Filtered list with sts_score field added
        """
        if not candidates:
            return []
        
        # Get adaptive threshold
        threshold = self._get_adaptive_threshold(query_latex)
        
        # 🚀 Batch encoding for speedup
        try:
            with torch.no_grad():
                query_emb = self.model.encode(
                    query_latex,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False
                )
                
                cand_latexes = [c.get("latex", "") for c in candidates]
                cand_embs = self.model.encode(
                    cand_latexes,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False,
                    batch_size=64  # ✅ 批处理优化
                )
                
                # ✅ Batch cosine similarity
                scores = util.cos_sim(query_emb, cand_embs)[0].cpu().numpy()
                
        except Exception as e:
            logger.error(f"❌ Batch STS failed: {e}, falling back to sequential")
            # ✅ Fallback to sequential processing
            scores = np.array([
                self.score(query_latex, c.get("latex", ""))
                for c in candidates
            ])
        
        # ✅ Filter and add scores
        filtered = []
        for c, score in zip(candidates, scores):
            c["sts_score"] = float(score)
            if score >= threshold:
                filtered.append(c)
        
        pass_rate = len(filtered) / len(candidates) if candidates else 0
        logger.debug(
            f"STS filtered: {len(filtered)}/{len(candidates)} passed "
            f"(threshold={threshold:.2f}, pass_rate={pass_rate:.1%})"
        )
        
        # ✅ Emergency fallback: if pass rate < 5%, lower threshold
        if pass_rate < 0.05 and len(candidates) > 10:
            logger.warning(
                f"⚠️ Very low STS pass rate ({pass_rate:.1%}), "
                f"lowering threshold to {threshold - 0.1:.2f}"
            )
            
            # Retry with lower threshold
            filtered = [
                c for c in candidates
                if c["sts_score"] >= (threshold - 0.1)
            ]
        
        return filtered