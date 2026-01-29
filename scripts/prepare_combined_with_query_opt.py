import json
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
import re
from tqdm import tqdm

# ✅ 调大字段限制
csv.field_size_limit(sys.maxsize)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_mathml(xml_str):
    if not xml_str: return ""
    xml_str = re.sub(r'\s+[a-z]+="[^"]+"', '', xml_str)
    tags = re.findall(r'<([a-zA-Z0-9]+)', xml_str)
    ignored_tags = {'math', 'semantics', 'annotation', 'mstyle', 'mrow'}
    return ",".join([t for t in tags if t.lower() not in ignored_tags])

def normalize_visual_id(vid):
    """将 q_6 统一转为 6"""
    if not vid: return ""
    return str(vid).lower().replace('q_', '').strip()

def process_arqmath_data(corpus_shards=5):
    data_dir = Path("data/arqmath3")
    xml_path = data_dir / "Topics_Task2_2022_V0.1.xml"
    opt_dir = data_dir / "opt_representation_v3"
    latex_dir = data_dir / "latex_representation_v3"
    
    # 1. 解析 XML 映射 (B.301 -> q_6)
    tree = ET.parse(xml_path)
    qid_to_target_vid = {}
    qid_to_latex = {}
    for topic in tree.getroot().findall('.//Topic'):
        qid = topic.get('number')
        vid = topic.find('.//Formula_Id').text
        latex = topic.find('.//Latex').text
        qid_to_target_vid[qid] = normalize_visual_id(vid)
        qid_to_latex[qid] = latex.strip() if latex else ""
    
    target_vids = set(qid_to_target_vid.values())
    query_mathml_map = {}
    formulas_corpus = {}
    
    opt_files = sorted(opt_dir.glob("*.tsv"))
    latex_files = sorted(latex_dir.glob("*.tsv"))

    # 🚀 第一阶段：扫描所有 101 个分片，仅为捕获查询公式的 MathML (通过 visual_id)
    logger.info("🔎 正在全量扫描 101 个分片以捕获查询公式的结构...")
    for f_path in tqdm(opt_files, desc="Scanning for queries"):
        with open(f_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) >= 9:
                    row_vid = normalize_visual_id(row[6]) # 第七列 visual_id
                    if row_vid in target_vids:
                        skel = clean_mathml(row[8])
                        for qid, t_vid in qid_to_target_vid.items():
                            if row_vid == t_vid:
                                query_mathml_map[qid] = skel
        if len(query_mathml_map) == len(qid_to_target_vid): break

    # 🚀 第二阶段：构建语料库 (通过 id 列作为 Key)
    logger.info(f"📦 正在构建前 {corpus_shards} 个分片的语料库...")
    for i in range(min(corpus_shards, len(opt_files))):
        o_path = opt_files[i]
        l_path = latex_files[i]
        
        # 处理 OPT (Key 是 id)
        with open(o_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) >= 9:
                    fid = row[0].strip() # ✅ 第一列 id
                    formulas_corpus[fid] = {"formula_id": fid, "mathml_skel": clean_mathml(row[8])}

        # 处理 LaTeX (Key 是 id)
        with open(l_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) >= 9:
                    fid = row[0].strip() # ✅ 第一列 id
                    if fid in formulas_corpus:
                        formulas_corpus[fid]["latex"] = row[8].strip()

    # 保存结果
    out_dir = Path("data/processed")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "formulas.json", 'w') as f:
        json.dump(formulas_corpus, f, indent=2)
    
    q_full = {qid: {"query_id": qid, "latex": qid_to_latex[qid], "mathml_skel": query_mathml_map.get(qid, "")} for qid in qid_to_latex}
    with open(out_dir / "queries_full.json", 'w') as f:
        json.dump(q_full, f, indent=2)
    
    with open(out_dir / "queries.json", 'w') as f:
        json.dump({qid: data["latex"] for qid, data in q_full.items()}, f, indent=2)

    logger.info(f"✅ 完成！捕获率: {len(query_mathml_map)}/100, 语料库: {len(formulas_corpus)} 条")

if __name__ == "__main__":
    process_arqmath_data(corpus_shards=5)
