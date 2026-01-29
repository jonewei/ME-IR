"""
从ARQMath Topics XML文件中提取查询公式(修复版)
适配ARQMath的特殊XML格式:
- Topic number是属性而非子标签
- 公式以LaTeX格式存储
- 需要转换LaTeX为MathML骨架
"""

import xml.etree.ElementTree as ET
import json
import re
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 🚀 核心1: LaTeX清理与标准化
# ============================================================
def clean_latex_from_html(latex_str):
    """
    清理从HTML中提取的LaTeX公式
    """
    if not latex_str:
        return ""
    
    # 移除HTML实体
    latex_str = re.sub(r'&lt;', '<', latex_str)
    latex_str = re.sub(r'&gt;', '>', latex_str)
    latex_str = re.sub(r'&quot;', '"', latex_str)
    latex_str = re.sub(r'&amp;', '&', latex_str)
    
    # 移除LaTeX包裹符号
    latex_str = re.sub(r'^\$+|\$+$', '', latex_str.strip())
    latex_str = re.sub(r'^\\begin\{equation\*?\}|\\end\{equation\*?\}$', '', latex_str)
    latex_str = re.sub(r'^\\begin\{align\*?\}|\\end\{align\*?\}$', '', latex_str)
    
    # 统一空格
    latex_str = re.sub(r'\s+', ' ', latex_str.strip())
    
    return latex_str

def normalize_latex_for_matching(latex_str):
    """
    LaTeX标准化(与prepare_final_arqmath.py保持一致)
    """
    if not latex_str:
        return ""
    
    latex_str = clean_latex_from_html(latex_str)
    
    # 统一排版差异
    latex_str = re.sub(r'\\dfrac', r'\\frac', latex_str)
    latex_str = re.sub(r'\\tfrac', r'\\frac', latex_str)
    latex_str = re.sub(r'\\left|\\right', '', latex_str)
    latex_str = re.sub(r'\\cdot', r'\\times', latex_str)
    latex_str = re.sub(r'\|\|', r'\\|', latex_str)
    
    return latex_str.lower()

def latex_to_pseudo_mathml(latex_str):
    """
    将LaTeX转换为伪MathML骨架
    策略:提取关键数学结构而非渲染
    """
    if not latex_str:
        return ""
    
    # 清理
    latex = clean_latex_from_html(latex_str)
    
    # 提取数学结构关键词
    structure_tags = []
    
    # 1. 分数
    if r'\frac' in latex:
        structure_tags.append('mfrac')
    
    # 2. 根号
    if r'\sqrt' in latex:
        structure_tags.append('msqrt')
    
    # 3. 上下标
    if '^' in latex:
        structure_tags.append('msup')
    if '_' in latex:
        structure_tags.append('msub')
    
    # 4. 积分/求和/极限
    if r'\int' in latex:
        structure_tags.append('mo,mo')  # integral operator
    if r'\sum' in latex:
        structure_tags.append('mo,mo')
    if r'\lim' in latex:
        structure_tags.append('mo')
    
    # 5. 矩阵
    if r'\begin{' in latex and ('matrix' in latex or 'bmatrix' in latex):
        structure_tags.append('mtable,mtr,mtd')
    
    # 6. 括号
    paren_count = latex.count('(') + latex.count('[') + latex.count(r'\{')
    if paren_count > 0:
        structure_tags.extend(['mo'] * min(paren_count, 3))
    
    # 7. 运算符
    operators = [r'\times', r'\div', '+', '-', '=', r'\leq', r'\geq', r'\in']
    for op in operators:
        if op in latex:
            structure_tags.append('mo')
    
    # 8. 数字和标识符
    if re.search(r'\d', latex):
        structure_tags.append('mn')
    if re.search(r'[a-zA-Z]', latex):
        structure_tags.append('mi')
    
    return ','.join(structure_tags) if structure_tags else ""

# ============================================================
# 🚀 核心2: XML解析器(修复版)
# ============================================================
def extract_formulas_from_html(html_str):
    """
    从HTML字符串中提取所有math-container公式
    """
    if not html_str:
        return []
    
    # 匹配 <span class="math-container" id="q_X">$...$</span>
    pattern = r'<span class="math-container" id="(q_\d+)">(.*?)</span>'
    matches = re.findall(pattern, html_str, re.DOTALL)
    
    formulas = []
    for formula_id, latex_content in matches:
        cleaned = clean_latex_from_html(latex_content)
        if cleaned:
            formulas.append((formula_id, cleaned))
    
    return formulas

def parse_arqmath_topics_xml(xml_file):
    """
    解析ARQMath Topics XML文件(适配实际格式)
    """
    logger.info(f"📂 Parsing XML file: {xml_file}")
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"❌ Failed to parse XML: {e}")
        return {}
    
    queries = {}
    
    for topic in root.findall('.//Topic'):
        try:
            # 🚀 修复1: Topic number是属性
            topic_number = topic.get('number')
            if not topic_number:
                logger.warning("  ⚠️ Topic missing 'number' attribute, skipping")
                continue
            
            # 🚀 修复2: 安全获取子标签
            formula_id_elem = topic.find('Formula_Id')
            latex_elem = topic.find('Latex')
            title_elem = topic.find('Title')
            question_elem = topic.find('Question')
            tags_elem = topic.find('Tags')
            
            # 提取文本(带默认值)
            formula_id = formula_id_elem.text.strip() if formula_id_elem is not None and formula_id_elem.text else ""
            main_latex = latex_elem.text.strip() if latex_elem is not None and latex_elem.text else ""
            title = title_elem.text if title_elem is not None else ""
            question = ET.tostring(question_elem, encoding='unicode', method='html') if question_elem is not None else ""
            tags = tags_elem.text.strip() if tags_elem is not None and tags_elem.text else ""
            
            # 🚀 核心3: 提取所有公式
            all_formulas = []
            
            # 主公式(来自<Latex>标签)
            if main_latex:
                all_formulas.append(('main', formula_id, main_latex))
            
            # 标题中的公式
            if title:
                title_formulas = extract_formulas_from_html(title)
                all_formulas.extend([('title', fid, latex) for fid, latex in title_formulas])
            
            # 问题中的公式
            if question:
                question_formulas = extract_formulas_from_html(question)
                all_formulas.extend([('question', fid, latex) for fid, latex in question_formulas])
            
            # 🚀 核心4: 选择主查询公式(使用<Latex>标签的公式)
            primary_latex = main_latex if main_latex else (all_formulas[0][2] if all_formulas else "")
            
            # 构建查询对象
            queries[topic_number] = {
                'query_id': topic_number,
                'formula_id': formula_id,
                'latex': primary_latex,
                'latex_norm': normalize_latex_for_matching(primary_latex),
                'mathml_skel': latex_to_pseudo_mathml(primary_latex),  # 🚀 伪MathML骨架
                'title': title,
                'tags': tags,
                'num_formulas': len(all_formulas),
                'all_formulas': all_formulas
            }
            
        except Exception as e:
            logger.warning(f"  ⚠️ Error parsing topic {topic.get('number', 'unknown')}: {e}")
    
    logger.info(f"✅ Parsed {len(queries)} topics")
    
    return queries

# ============================================================
# 🚀 核心3: 从corpus匹配真实MathML(补充策略)
# ============================================================
def match_real_mathml_from_corpus(queries, corpus_file):
    """
    尝试从corpus中匹配真实的MathML骨架
    """
    logger.info("🔍 Matching real MathML from corpus...")
    
    if not Path(corpus_file).exists():
        logger.warning(f"⚠️ Corpus file not found: {corpus_file}")
        logger.warning("   Using pseudo-MathML only")
        return queries
    
    # 加载corpus
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    # 构建反向索引: normalized_latex -> mathml_skel
    latex_to_mathml = {}
    for fid, formula in corpus.items():
        norm_latex = formula.get('latex_norm', '')
        mathml_skel = formula.get('mathml_skel', '')
        
        if norm_latex and mathml_skel:
            if norm_latex not in latex_to_mathml:
                latex_to_mathml[norm_latex] = mathml_skel
    
    logger.info(f"  Built index with {len(latex_to_mathml)} normalized LaTeX")
    
    # 匹配查询
    matched = 0
    for qid, qdata in queries.items():
        norm_latex = qdata['latex_norm']
        
        if norm_latex in latex_to_mathml:
            # 找到精确匹配,用真实MathML替换伪MathML
            qdata['mathml_skel'] = latex_to_mathml[norm_latex]
            qdata['mathml_source'] = 'corpus_exact'
            matched += 1
        else:
            qdata['mathml_source'] = 'pseudo_mathml'
    
    logger.info(f"  Matched real MathML for {matched}/{len(queries)} queries ({matched/len(queries)*100:.1f}%)")
    logger.info(f"  Using pseudo-MathML for {len(queries)-matched} queries")
    
    return queries

# ============================================================
# 🚀 核心4: 质量验证
# ============================================================
def validate_query_quality(queries):
    """
    验证查询数据的完整性
    """
    logger.info("🔍 Validating query quality...")
    
    stats = {
        'total': len(queries),
        'with_latex': 0,
        'with_mathml': 0,
        'with_both': 0,
        'with_multiple_formulas': 0,
        'real_mathml': 0,
        'pseudo_mathml': 0,
        'incomplete': []
    }
    
    for qid, qdata in queries.items():
        has_latex = bool(qdata.get('latex'))
        has_mathml = bool(qdata.get('mathml_skel'))
        
        if has_latex:
            stats['with_latex'] += 1
        if has_mathml:
            stats['with_mathml'] += 1
        if has_latex and has_mathml:
            stats['with_both'] += 1
        
        if qdata.get('num_formulas', 0) > 1:
            stats['with_multiple_formulas'] += 1
        
        # 统计MathML来源
        if qdata.get('mathml_source') == 'corpus_exact':
            stats['real_mathml'] += 1
        elif qdata.get('mathml_source') == 'pseudo_mathml':
            stats['pseudo_mathml'] += 1
        
        if not has_latex and not has_mathml:
            stats['incomplete'].append(qid)
    
    logger.info("="*60)
    logger.info("📊 Query Quality Report:")
    logger.info(f"  Total queries: {stats['total']}")
    logger.info(f"  With LaTeX: {stats['with_latex']} ({stats['with_latex']/stats['total']*100:.1f}%)")
    logger.info(f"  With MathML skeleton: {stats['with_mathml']} ({stats['with_mathml']/stats['total']*100:.1f}%)")
    logger.info(f"  - Real MathML (from corpus): {stats['real_mathml']}")
    logger.info(f"  - Pseudo MathML (from LaTeX): {stats['pseudo_mathml']}")
    logger.info(f"  Multi-formula queries: {stats['with_multiple_formulas']}")
    
    if stats['incomplete']:
        logger.warning(f"  ⚠️ Incomplete queries: {len(stats['incomplete'])}")
        logger.warning(f"    Sample: {stats['incomplete'][:5]}")
    
    logger.info("="*60)
    
    return stats

# ============================================================
# 🚀 主流程
# ============================================================
def main():
    # 路径配置
    xml_file = Path("data/arqmath3/Topics_Task2_2022_V0.1.xml")
    corpus_file = Path("data/processed/formulas.json")
    output_file = Path("data/processed/queries_full_with_mathml.json")
    
    if not xml_file.exists():
        logger.error(f"❌ XML file not found: {xml_file}")
        return
    
    # Step 1: 解析XML
    queries = parse_arqmath_topics_xml(xml_file)
    
    if not queries:
        logger.error("❌ Failed to parse XML. Aborting.")
        return
    
    # 统计公式分布
    formula_counts = [q['num_formulas'] for q in queries.values()]
    logger.info(f"📊 Formula distribution:")
    logger.info(f"  Queries with formulas: {sum(1 for c in formula_counts if c > 0)}/{len(formula_counts)}")
    logger.info(f"  Total formulas extracted: {sum(formula_counts)}")
    logger.info(f"  Avg formulas per query: {sum(formula_counts)/len(formula_counts):.2f}")
    
    # Step 2: 从corpus匹配真实MathML(可选)
    queries = match_real_mathml_from_corpus(queries, corpus_file)
    
    # Step 3: 质量验证
    stats = validate_query_quality(queries)
    
    # Step 4: 保存结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Saved enhanced queries to {output_file}")
    
    # Step 5: 生成诊断报告
    diagnostic_report = {
        'xml_file': str(xml_file),
        'output_file': str(output_file),
        'statistics': stats,
        'sample_queries': {
            qid: {
                'query_id': qdata['query_id'],
                'latex': qdata.get('latex', '')[:100],
                'mathml_skel': qdata.get('mathml_skel', ''),
                'mathml_source': qdata.get('mathml_source', 'unknown'),
                'num_formulas': qdata.get('num_formulas', 0)
            }
            for qid, qdata in list(queries.items())[:5]
        }
    }
    
    report_file = output_file.parent / "mathml_extraction_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(diagnostic_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Diagnostic report saved to {report_file}")
    
    # 🎯 关键提示
    if stats['pseudo_mathml'] > stats['real_mathml']:
        logger.warning("="*60)
        logger.warning("⚠️ NOTICE: Majority of queries use pseudo-MathML")
        logger.warning("   Pseudo-MathML is derived from LaTeX structure")
        logger.warning("   It may have lower matching precision than real MathML")
        logger.warning("   Recommendation: Run fix_query_mathml_matching.py to improve coverage")
        logger.warning("="*60)

if __name__ == "__main__":
    main()