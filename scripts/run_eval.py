"""
Enhanced evaluation script for ARQMath-3 (Fixed imports)

Key fixes:
1. ✅ Corrected all import paths
2. ✅ Added missing module checks
3. ✅ Better error handling for missing dependencies
"""

import pickle
import json
import yaml
import os
import logging
import argparse
import sys
from datetime import datetime
from pathlib import Path

# ✅ Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ✅ Fixed imports with proper paths
from retrieval.recall_api import StructuralRecall
from retrieval.approach0_hash import Approach0HashIndex
from ranking.mathbert_embedder import MathBERTEmbedder
from ranking.feature_builder import FeatureBuilder
from ranking.lgbm_ranker import LambdaRanker
from ranking.coarse_rank import CoarseRanker
from filtering.formula_sts_model import FormulaSTSModel
from filtering.high_conf_filter import HighConfidenceFilter
from evaluation.eval_runner import evaluate, save_trec_run, load_qrel_labels

# ✅ Optional: Graph reranking (may not exist yet)
try:
    from graph.graph_prior import GraphPrior
    GRAPH_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Graph reranking not available, skipping Stage 4")
    GRAPH_AVAILABLE = False

# ✅ Optional: SearchPipeline
try:
    from pipeline.search_pipeline import SearchPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ SearchPipeline not found, will use manual orchestration")
    PIPELINE_AVAILABLE = False


# ========== Logging Setup ==========
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f"logs/eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_experiment_config(path):
    """Load YAML configuration"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_formulas(path):
    """Load and standardize formula data"""
    logger.info(f"📂 Loading formulas from {path}")
    with open(path) as f:
        raw_formulas = json.load(f)
    
    formulas = {}
    for fid, formula in raw_formulas.items():
        if isinstance(formula, str):
            # Legacy format (string only)
            formulas[fid] = {
                "formula_id": fid,
                "latex": formula,
                "mathml_skel": "",
                "depth": 0,
                "graph_degree": {"in": 0, "out": 0}
            }
        else:
            # Modern format (dict)
            formulas[fid] = dict(formula)
            formulas[fid]["formula_id"] = fid
    
    logger.info(f"✅ Loaded {len(formulas):,} formulas")
    return formulas


def load_or_build_index(formulas, index_path):
    """Load or build hash index"""
    if os.path.exists(index_path):
        logger.info(f"📂 Loading pre-built index from {index_path}")
        index = Approach0HashIndex()
        index.load(index_path)
        
        # ✅ Show index statistics
        report = index.collision_report()
        logger.info(f"   Total buckets: {report['total_buckets']:,}")
        logger.info(f"   Total formulas: {report['total_formulas']:,}")
        logger.info(f"   Avg bucket size: {report['avg_bucket_size']:.2f}")
        logger.info(f"   Collision rate: {report['collision_rate']:.1%}")
        
    else:
        logger.info("🔧 Building new index...")
        logger.warning("   This may take 10-30 minutes for 28M formulas")
        
        from tqdm import tqdm
        index = Approach0HashIndex()
        
        for fid, formula in tqdm(formulas.items(), desc="Indexing"):
            index.add(formula)
        
        # Save for future use
        index.save(index_path)
        logger.info(f"💾 Index saved to {index_path}")
    
    return index


def build_pipeline_manual(config, index, formulas):
    """
    Build pipeline manually (fallback when SearchPipeline unavailable)
    """
    logger.info("🔧 Building pipeline components manually...")
    
    # Stage 1: Structural Recall
    recall = StructuralRecall(
        index,
        enable_fuzzy=True,  # ✅ Enable fuzzy search
        fuzzy_max_distance=2,
        fuzzy_max_buckets=50
    )
    
    # Stage 2: Coarse Ranking
    embedder = MathBERTEmbedder(model_name=config['models']['mathbert'])
    fb = FeatureBuilder(
        num_features=config['features']['num_features'],
        normalize=config['features'].get('normalize', False)
    )
    ranker = LambdaRanker(model_path=config['models']['ranker'])
    coarse = CoarseRanker(embedder, ranker, fb)
    
    # Stage 3: STS Filtering
    sts = FormulaSTSModel(
        model_name=config['models']['sts'],
        threshold=config['pipeline']['sts_threshold'],
        use_adaptive_threshold=True  # ✅ Enable adaptive thresholding
    )
    filterer = HighConfidenceFilter(sts)
    
    # Stage 4: Graph Reranking (optional)
    graph_prior = None
    if GRAPH_AVAILABLE and os.path.exists(config['data']['pagerank']):
        with open(config['data']['pagerank'], "rb") as f:
            pr = pickle.load(f)
        graph_prior = GraphPrior(pr, alpha=config['pipeline'].get('derive_alpha', 0.3))
    
    # Return components dict (for manual orchestration)
    return {
        'recall': recall,
        'coarse': coarse,
        'filterer': filterer,
        'graph_prior': graph_prior,
        'config': config
    }


def search_manual(components, query):
    """
    Manual 4-stage search (when SearchPipeline unavailable)
    """
    config = components['config']
    
    # Stage 1: Recall
    candidates = components['recall'].recall(
        query,
        topk=config['pipeline']['topk_recall']
    )
    
    if not candidates:
        logger.warning(f"Query {query.get('query_id')}: Zero recall")
        return []
    
    # Stage 2: Coarse Ranking
    candidates = components['coarse'].rank(query, candidates)
    candidates = candidates[:config['pipeline']['topk_rank']]
    
    if not candidates:
        logger.warning(f"Query {query.get('query_id')}: Zero after ranking")
        return []
    
    # Stage 3: STS Filtering
    candidates = components['filterer'].apply(query, candidates)
    
    if not candidates:
        logger.warning(f"Query {query.get('query_id')}: Zero after filtering")
        # ✅ Fallback: return coarse ranking results
        candidates = components['coarse'].rank(query, 
            components['recall'].recall(query, topk=config['pipeline']['topk_recall'])
        )[:config['pipeline']['final_k']]
    
    # Stage 4: Graph Reranking (optional)
    if components['graph_prior']:
        candidates = components['graph_prior'].rerank(candidates)
    
    # Final Top-K
    return candidates[:config['pipeline']['final_k']]


def main():
    parser = argparse.ArgumentParser(description="ARQMath Evaluation")
    parser.add_argument(
        "--config",
        default="experiments/exp_arqmath3.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit queries for testing"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🚀 Mathematical Expression Retrieval Evaluation")
    logger.info("=" * 60)

    # Load config and data
    config = load_experiment_config(args.config)
    formulas = load_formulas(config['data']['formulas'])
    index = load_or_build_index(formulas, config['data']['index'])
    
    # Build pipeline
    if PIPELINE_AVAILABLE:
        logger.info("✅ Using integrated SearchPipeline")
        # Use original build_pipeline logic
        # ...
    else:
        logger.info("⚠️ Using manual pipeline orchestration")
        components = build_pipeline_manual(config, index, formulas)
    
    # Load queries
    queries_path = config['data']['queries']
    queries_full_path = queries_path.replace('queries.json', 'queries_final.json')
    
    if Path(queries_full_path).exists():
        logger.info(f"📂 Loading enhanced queries from {queries_full_path}")
        with open(queries_full_path) as f:
            q_dict = json.load(f)
        all_queries = list(q_dict.values())
    else:
        logger.info(f"📂 Loading basic queries from {queries_path}")
        with open(queries_path) as f:
            q_dict = json.load(f)
        all_queries = [{"query_id": qid, "latex": text} for qid, text in q_dict.items()]
    
    # Limit queries if needed
    queries = all_queries[:args.max_queries] if args.max_queries else all_queries
    logger.info(f"🔍 Processing {len(queries)} queries")
    
    # Load labels
    labels = load_qrel_labels(config['data']['labels'])
    
    # Run evaluation
    logger.info("\n" + "-" * 30)
    logger.info("🔍 EVALUATION IN PROGRESS")
    logger.info("-" * 30)
    
    # ✅ Manual evaluation loop if SearchPipeline unavailable
    if not PIPELINE_AVAILABLE:
        from tqdm import tqdm
        all_results = {}
        
        for query in tqdm(queries, desc="Evaluating"):
            try:
                results = search_manual(components, query)
                all_results[query["query_id"]] = results
            except Exception as e:
                logger.error(f"Query {query.get('query_id')} failed: {e}")
                all_results[query["query_id"]] = []
        
        # Calculate metrics manually
        from evaluation.eval_runner import calculate_metrics
        metrics = calculate_metrics(all_results, labels)
    else:
        # Use integrated evaluation
        metrics, all_results = evaluate(pipeline, queries, labels)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    save_trec_run(all_results, str(output_dir / "run.trec"))
    
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Evaluation Complete!")
    logger.info(f"📊 Results: {output_dir}")
    logger.info(f"📈 Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.4f}")
        else:
            logger.info(f"   {key}: {value}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()