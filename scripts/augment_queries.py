import json
import logging
import latex2mathml.converter
from pathlib import Path
from tqdm import tqdm
import re
import xml.etree.ElementTree as ET

# 导入你之前定义的 DNA 提取逻辑
# 注意：这里直接复用之前的函数，确保 DNA 格式一致
def clean_mathml_to_dna(xml_str):
    if not xml_str: return ""
    # 清理命名空间
    xml_str = re.sub(r'\s+xmlns="[^"]+"|\s+encoding="[^"]+"', '', xml_str)
    IGNORED = {'math', 'semantics', 'annotation', 'annotation-xml', 'mstyle', 'mrow', 'mtext', 'mspace'}
    
    def get_structure(element):
        tag = element.tag.split('}')[-1].lower()
        if tag in IGNORED:
            return "".join([get_structure(child) for child in element])
        if tag in {'ci', 'cn', 'mi', 'mn'}: return "v" # 增加 mi, mn 适配 Presentation MathML
        
        children = [get_structure(child) for child in element]
        children = [c for c in children if c]
        return f"{tag}[{','.join(children)}]" if children else tag

    try:
        tree = ET.fromstring(xml_str)
        return get_structure(tree)
    except:
        return ""

def augment_queries():
    query_path = Path("data/processed/queries_full.json")
    if not query_path.exists():
        print("❌ 错误：找不到 queries_full.json，请先运行 prepare 脚本。")
        return

    with open(query_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    print(f"🚀 开始补全 {len(queries)} 条查询的 MathML...")
    
    converted_count = 0
    for qid, qdata in tqdm(queries.items(), desc="Converting LaTeX to DNA"):
        # 只有在 MathML 缺失时才转换
        if not qdata.get('mathml_skel'):
            try:
                # 1. LaTeX -> MathML (Presentation MathML)
                mathml_output = latex2mathml.converter.convert(qdata['latex'])
                
                # 2. MathML -> Structural DNA
                dna = clean_mathml_to_dna(mathml_output)
                
                if dna:
                    qdata['mathml_skel'] = dna
                    converted_count += 1
            except Exception as e:
                # 针对一些复杂的 LaTeX 语法可能转换失败，跳过即可
                continue

    # 保存更新后的查询文件
    with open(query_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    print(f"✅ 补全完成！成功转换了 {converted_count} 条查询。")
    print(f"📈 现在的查询 MathML 覆盖率: {(sum(1 for q in queries.values() if q['mathml_skel'])/len(queries)):.1%}")

if __name__ == "__main__":
    augment_queries()