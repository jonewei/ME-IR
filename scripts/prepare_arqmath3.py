import json
import csv
import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from retrieval.approach0_hash import DualHashGenerator
from retrieval.indexer import FormulaIndexer

csv.field_size_limit(sys.maxsize)

def normalize_latex(latex_str):
    if not latex_str: return ""
    latex_str = re.sub(r'\s+', ' ', latex_str.strip())
    latex_str = re.sub(r'\\dfrac|\\tfrac', r'\\frac', latex_str)
    latex_str = re.sub(r'\\left|\\right', '', latex_str)
    return latex_str.lower()

def clean_mathml_to_dna(xml_str):
    """DFS 提取结构化 DNA"""
    if not xml_str: return ""
    xml_str = re.sub(r'\s+xmlns="[^"]+"|\s+encoding="[^"]+"', '', xml_str)
    IGNORED = {'math', 'semantics', 'annotation', 'annotation-xml', 'mstyle', 'mrow', 'mtext', 'mspace'}
    
    def get_structure(element):
        tag = element.tag.split('}')[-1].lower()
        if tag in IGNORED:
            return "".join([get_structure(child) for child in element])
        if tag in {'ci', 'cn'}: return "v"
        
        children = [get_structure(child) for child in element]
        children = [c for c in children if c]
        return f"{tag}[{','.join(children)}]" if children else tag

    try:
        tree = ET.fromstring(xml_str)
        return get_structure(tree)
    except:
        return ""

def process_all(num_shards=10):
    data_dir = Path("data/arqmath3")
    opt_dir = data_dir / "opt_representation_v3"
    latex_dir = data_dir / "latex_representation_v3"
    
    indexer = FormulaIndexer()
    hash_gen = DualHashGenerator()
    
    # 1. 加载并对齐数据
    corpus = {}
    latex_files = sorted(Path(latex_dir).glob("*.tsv"))[:num_shards]
    for f in tqdm(latex_files, desc="Processing LaTeX"):
        with open(f, 'r', encoding='utf-8') as fin:
            reader = csv.reader(fin, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) >= 9:
                    fid, l_raw = row[0].strip(), row[8].strip()
                    corpus[fid] = {'latex_norm': normalize_latex(l_raw)}

    # 2. 提取 DNA 并构建索引
    opt_files = sorted(Path(opt_dir).glob("*.tsv"))[:num_shards]
    batch = []
    for f in tqdm(opt_files, desc="Building DNA & Index"):
        with open(f, 'r', encoding='utf-8') as fin:
            reader = csv.reader(fin, delimiter='\t')
            next(reader)
            for row in reader:
                fid = row[0].strip()
                if fid in corpus:
                    dna = clean_mathml_to_dna(row[8])
                    corpus[fid]['mathml_skel'] = dna
                    h = hash_gen.get_dual_hash(corpus[fid]['latex_norm'], dna)
                    batch.append((fid, h['h_latex'], h['h_dna']))
                    if len(batch) >= 10000:
                        indexer.save_batch(batch)
                        batch = []
    if batch: indexer.save_batch(batch)

    # 保存 formulas.json 供后续阶段使用
    out_dir = Path("data/processed")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "formulas.json", 'w') as f:
        json.dump(corpus, f, indent=2)
    print("✅ 索引构建完成！")

if __name__ == "__main__":
    process_all(num_shards=10) # 建议先用 10 个分片测试