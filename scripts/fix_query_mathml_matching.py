"""
匹配补充脚本:为XML中缺失MathML的查询提供备选方案
使用多种匹配策略确保覆盖率
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 🚀 核心1: 多策略LaTeX标准化
# ============================================================
def normalize_latex_aggressive(latex_str):
    """
    激进的LaTeX标准化(容忍更多变体)
    """
    if not latex_str:
        return ""
    
    # 1. 基础清理
    latex_str = re.sub(r'\s+', ' ', latex_str.strip())
    
    # 2. 统一分数表示
    latex_str = re.sub(r'\\dfrac', r'\\frac', latex_str)
    latex_str = re.sub(r'\\tfrac', r'\\frac', latex_str)
    latex_str = re.sub(r'\\cfrac', r'\\frac', latex_str)
    
    # 3. 移除修饰符
    latex_str = re.sub(r'\\left|\\right', '', latex_str)
    latex_str = re.sub(r'\\big|\\Big|\\bigg|\\Bigg', '', latex_str)
    
    # 4. 统一运算符
    latex_str = re.sub(r'\\cdot', r'\\times', latex_str)
    latex_str = re.sub(r'\\ast', r'\\times', latex_str)
    
    # 5. 统一括号
    latex_str = re.sub(r'\\\(', '(', latex_str)
    latex_str = re.sub(r'\\\)', ')', latex_str)
    latex_str = re.sub(r'\\\[', '[', latex_str)
    latex_str = re.sub(r'\\\]', ']', latex_str)
    
    # 6. 统一范数/绝对值
    latex_str = re.sub(r'\|\|', r'\\|', latex_str)
    
    # 7. 移除多余空格和花括号
    latex_str = re.sub(r'\{\s*(\w)\s*\}', r'\1', latex_str)
    
    return latex_str.lower()

def compute_latex_similarity(latex1, latex2):
    """
    计算两个LaTeX字符串的相似度(0-1)
    """
    norm1 = normalize_latex_aggressive(latex1)
    norm2 = normalize_latex_aggressive(latex2)
    
    return SequenceMatcher(None, norm1, norm2).ratio()

# ============================================================
# 🚀 核心2: 构建语料库反向索引
# ============================================================
def build_corpus_reverse_index(corpus_file):
    """
    构建多种反向索引以支持不同匹配策略
    """
    logger.info(f"📂 Building reverse index from {corpus_file}...")
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    # 索引1: 精确标准化LaTeX -> MathML
    exact_index = {}
    
    # 索引2: LaTeX关键token -> [(formula_id, mathml_skel), ...]
    token_index = {}
    
    # 索引3: MathML骨架 -> LaTeX (用于反向验证)
    mathml_index = {}
    
    for fid, formula in corpus.items():
        latex = formula.get('latex', '')
        latex_norm = normalize_latex_aggressive(latex)
        mathml_skel = formula.get('mathml_skel', '')
        
        # 构建精确索引
        if latex_norm and mathml_skel:
            if latex_norm not in exact_index:
                exact_index[latex_norm] = mathml_skel
        
        # 构建token索引(提取数学符号)
        if latex and mathml_skel:
            tokens = re.findall(r'\\[a-zA-Z]+|[a-zA-Z0-9]+', latex)
            for token in tokens:
                if token not in token_index:
                    token_index[token] = []
                token_index[token].append((fid, mathml_skel))
        
        # 构建MathML索引
        if mathml_skel and latex:
            if mathml_skel not in mathml_index:
                mathml_index[mathml_skel] = []
            mathml_index[mathml_skel].append(latex)
    
    logger.info(f"  Exact index: {len(exact_index)} entries")
    logger.info(f"  Token index: {len(token_index)} tokens")
    logger.info(f"  MathML index: {len(mathml_index)} skeletons")
    
    return {
        'exact': exact_index,
        'token': token_index,
        'mathml': mathml_index,
        'corpus': corpus
    }

# ============================================================
# 🚀 核心3: 多策略匹配器
# ============================================================
def match_query_mathml_multiway(query_data, index_bundle):
    """
    使用多种策略为查询匹配MathML
    返回: (mathml_skel, confidence, method)
    """
    latex = query_data.get('latex', '')
    
    if not latex:
        return None, 0.0, 'no_latex'
    
    latex_norm = normalize_latex_aggressive(latex)
    
    # 策略1: 精确匹配(置信度100%)
    if latex_norm in index_bundle['exact']:
        return index_bundle['exact'][latex_norm], 1.0, 'exact_match'
    
    # 策略2: 模糊匹配(基于编辑距离,置信度60-90%)
    best_match = None
    best_score = 0.0
    
    for candidate_latex, candidate_mathml in index_bundle['exact'].items():
        similarity = compute_latex_similarity(latex, candidate_latex)
        
        if similarity > best_score and similarity > 0.85:  # 阈值85%
            best_score = similarity
            best_match = candidate_mathml
    
    if best_match:
        return best_match, best_score, 'fuzzy_match'
    
    # 策略3: Token共现匹配(置信度40-70%)
    tokens = re.findall(r'\\[a-zA-Z]+|[a-zA-Z0-9]+', latex)
    
    if tokens:
        # 统计每个MathML出现的频率
        mathml_votes = {}
        
        for token in tokens:
            if token in index_bundle['token']:
                for fid, mathml_skel in index_bundle['token'][token]:
                    mathml_votes[mathml_skel] = mathml_votes.get(mathml_skel, 0) + 1
        
        if mathml_votes:
            # 选择得票最高的MathML
            best_mathml = max(mathml_votes, key=mathml_votes.get)
            vote_ratio = mathml_votes[best_mathml] / len(tokens)
            
            if vote_ratio > 0.5:  # 至少50%的token匹配
                return best_mathml, vote_ratio * 0.7, 'token_vote'
    
    # 策略4: 失败
    return None, 0.0, 'no_match'

# ============================================================
# 🚀 核心4: 批量补充
# ============================================================
def supplement_missing_mathml(queries_file, corpus_file, output_file):
    """
    为缺失MathML的查询进行补充
    """
    logger.info("🔄 Starting MathML supplementation...")
    
    # 加载数据
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    # 构建索引
    index_bundle = build_corpus_reverse_index(corpus_file)
    
    # 统计
    stats = {
        'total': len(queries),
        'already_has_mathml': 0,
        'exact_match': 0,
        'fuzzy_match': 0,
        'token_vote': 0,
        'no_match': 0,
        'confidence_distribution': []
    }
    
    # 处理每个查询
    for qid, qdata in queries.items():
        if qdata.get('mathml_skel'):
            stats['already_has_mathml'] += 1
            continue
        
        # 尝试匹配
        mathml_skel, confidence, method = match_query_mathml_multiway(qdata, index_bundle)
        
        if mathml_skel:
            qdata['mathml_skel'] = mathml_skel
            qdata['mathml_source'] = method
            qdata['mathml_confidence'] = confidence
            
            stats[method] += 1
            stats['confidence_distribution'].append(confidence)
        else:
            stats['no_match'] += 1
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    
    # 报告
    logger.info("="*60)
    logger.info("📊 Supplementation Report:")
    logger.info(f"  Total queries: {stats['total']}")
    logger.info(f"  Already had MathML: {stats['already_has_mathml']}")
    logger.info(f"  Exact matches: {stats['exact_match']}")
    logger.info(f"  Fuzzy matches: {stats['fuzzy_match']}")
    logger.info(f"  Token voting: {stats['token_vote']}")
    logger.info(f"  No match: {stats['no_match']}")
    
    if stats['confidence_distribution']:
        avg_conf = sum(stats['confidence_distribution']) / len(stats['confidence_distribution'])
        logger.info(f"  Avg confidence: {avg_conf:.3f}")
    
    total_matched = stats['exact_match'] + stats['fuzzy_match'] + stats['token_vote']
    coverage = (stats['already_has_mathml'] + total_matched) / stats['total'] * 100
    logger.info(f"  Final MathML coverage: {coverage:.1f}%")
    logger.info("="*60)
    
    return queries, stats

# ============================================================
# 主流程
# ============================================================
def main():
    queries_file = Path("data/processed/queries_full_with_mathml.json")
    corpus_file = Path("data/processed/formulas.json")
    output_file = Path("data/processed/queries_final.json")
    
    if not queries_file.exists():
        logger.error(f"❌ Queries file not found: {queries_file}")
        logger.error("   Run extract_query_mathml_from_xml.py first!")
        return
    
    if not corpus_file.exists():
        logger.error(f"❌ Corpus file not found: {corpus_file}")
        logger.error("   Run prepare_final_arqmath.py first!")
        return
    
    queries, stats = supplement_missing_mathml(queries_file, corpus_file, output_file)
    
    logger.info(f"✅ Final queries saved to {output_file}")
    
    # 保存统计报告
    report_file = output_file.parent / "supplementation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"📄 Report saved to {report_file}")

if __name__ == "__main__":
    main()