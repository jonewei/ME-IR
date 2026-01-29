"""
Structural Recall API with Fuzzy Matching

Key improvements:
1. ✅ Separated from hash implementation (clean architecture)
2. ✅ Added fuzzy search with Hamming distance
3. ✅ Performance optimization with early stopping
4. ✅ Fallback mechanism for zero-recall scenarios
"""

import logging
from typing import List, Dict, Any, Optional
from .approach0_hash import Approach0HashIndex, skeleton_hash

logger = logging.getLogger(__name__)


class StructuralRecall:
    """
    Stage 1: Structural Recall using Approach0 Hashing
    
    Features:
    - Exact hash matching (fast)
    - Fuzzy hash matching (fallback, with Hamming distance)
    - Automatic fallback to ensure non-zero recall
    """
    
    def __init__(
        self,
        index: Approach0HashIndex,
        enable_fuzzy: bool = True,
        fuzzy_max_distance: int = 2,
        fuzzy_max_buckets: int = 50
    ):
        """
        Args:
            index: Pre-built Approach0HashIndex
            enable_fuzzy: Enable fuzzy search as fallback
            fuzzy_max_distance: Max Hamming distance for fuzzy matching
            fuzzy_max_buckets: Max number of buckets to scan in fuzzy mode
        """
        self.index = index
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_max_distance = fuzzy_max_distance
        self.fuzzy_max_buckets = fuzzy_max_buckets
        
        logger.info(f"✅ StructuralRecall initialized")
        logger.info(f"   Index buckets: {len(index.index):,}")
        logger.info(f"   Fuzzy search: {enable_fuzzy}")
    
    def recall(
        self,
        query: Dict[str, Any],
        topk: int = 2000
    ) -> List[Dict]:
        """
        Retrieve candidates for a query.
        
        Args:
            query: Dict with 'latex' and optionally 'mathml_skel'
            topk: Maximum number of candidates to return
            
        Returns:
            List of candidate formula dicts
        """
        query_latex = query.get("latex", "")
        query_mathml = query.get("mathml_skel", "")
        
        if not query_latex:
            logger.warning("Query has no LaTeX, returning empty results")
            return []
        
        # Step 1: Exact matching (prioritize MathML if available)
        candidates = self._exact_match(query_latex, query_mathml)
        
        if len(candidates) >= topk:
            logger.debug(f"Exact match sufficient: {len(candidates)} candidates")
            return candidates[:topk]
        
        # Step 2: Fuzzy matching (if enabled and needed)
        if self.enable_fuzzy and len(candidates) < topk:
            logger.info(f"🔍 Exact match returned {len(candidates)} candidates, trying fuzzy search...")
            
            fuzzy_candidates = self._fuzzy_match(query_latex, query_mathml)
            
            # Merge and deduplicate
            seen_ids = {c['formula_id'] for c in candidates}
            for cand in fuzzy_candidates:
                if cand['formula_id'] not in seen_ids:
                    candidates.append(cand)
                    seen_ids.add(cand['formula_id'])
                    
                    if len(candidates) >= topk:
                        break
            
            logger.info(f"   After fuzzy: {len(candidates)} total candidates")
        
        # Step 3: Final fallback (return random sample if still zero)
        if len(candidates) == 0:
            logger.warning("⚠️ Zero recall! Falling back to random sample")
            candidates = self._fallback_random(topk)
        
        return candidates[:topk]
    
    def _exact_match(
        self,
        query_latex: str,
        query_mathml: Optional[str] = None
    ) -> List[Dict]:
        """Exact hash matching"""
        # Delegate to index's retrieve method
        return self.index.retrieve(
            query_latex=query_latex,
            mathml_skel=query_mathml
        )
    
    def _fuzzy_match(
        self,
        query_latex: str,
        query_mathml: Optional[str] = None
    ) -> List[Dict]:
        """
        Fuzzy matching using Hamming distance on hash values
        
        Strategy:
        1. Compute query hash
        2. Find buckets with similar hashes (Hamming distance <= threshold)
        3. Return candidates from these buckets
        """
        # Compute query hash
        if query_mathml:
            query_hash = skeleton_hash("", mathml_skel=query_mathml)
        else:
            query_hash = skeleton_hash(query_latex)
        
        # Convert hash to binary for Hamming distance
        query_hash_int = int(query_hash, 16)
        
        candidates = []
        bucket_count = 0
        
        # Scan index buckets
        for bucket_hash, bucket_candidates in self.index.index.items():
            # Compute Hamming distance
            bucket_hash_int = int(bucket_hash, 16)
            distance = bin(query_hash_int ^ bucket_hash_int).count('1')
            
            # Accept if within threshold
            if distance <= self.fuzzy_max_distance:
                candidates.extend(bucket_candidates)
                bucket_count += 1
                
                # Early stopping to avoid performance issues
                if bucket_count >= self.fuzzy_max_buckets:
                    logger.debug(f"Fuzzy search stopped at {bucket_count} buckets")
                    break
        
        logger.debug(f"Fuzzy matched {bucket_count} buckets, {len(candidates)} candidates")
        
        return candidates
    
    def _fallback_random(self, k: int = 1000) -> List[Dict]:
        """
        Emergency fallback: return random sample from corpus
        
        This ensures the system always returns something,
        which is important for evaluation continuity.
        """
        import random
        
        all_formulas = self.index.all_formulas
        
        if not all_formulas:
            logger.error("❌ Index is empty, cannot perform fallback")
            return []
        
        sample_size = min(k, len(all_formulas))
        return random.sample(all_formulas, sample_size)


# ============================================================
# Backward Compatibility Wrapper
# ============================================================
class Approach0Recall(StructuralRecall):
    """Alias for backward compatibility"""
    pass