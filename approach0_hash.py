# import re
# import hashlib
# import pickle
# from pathlib import Path

# class DualHashGenerator:
#     """
#     双重哈希生成器：
#     1. LaTeX 哈希：用于字面结构匹配
#     2. DNA 哈希：用于 MathML 拓扑结构匹配
#     """
#     def __init__(self):
#         # 预编译正则以提升千万级数据处理速度
#         self.delim_pattern = re.compile(r'\$\$?|\\\[|\\\]')
#         self.decor_pattern = re.compile(r'\\left|\\right|\\displaystyle|\\limits')
#         self.frac_pattern = re.compile(r'\\dfrac|\\tfrac')
#         self.space_pattern = re.compile(r'\s+')

#     def clean_latex(self, latex_str):
#         if not latex_str: return ""
#         # 1. 移除定界符
#         s = self.delim_pattern.sub('', latex_str)
        
#         # 2. 【新增】数学符号归一化
#         # 将 \| 替换为 ||
#         s = s.replace(r'\|', '||')
#         # 处理可能的共轭转置语义（可选，视你的研究目标而定）
#         # s = s.replace('^H', '^T') 
        
#         # 3. 移除装饰符
#         s = self.decor_pattern.sub('', s)
#         # 4. 标准化分式
#         s = self.frac_pattern.sub(r'\\frac', s)
#         # 5. 彻底删除所有空格
#         s = self.space_pattern.sub('', s.strip())
        
#         return s

#     def generate_latex_hash(self, clean_latex):
#         """生成 LaTeX 字符串的 MD5 指纹"""
#         if not clean_latex: return ""
#         return hashlib.md5(clean_latex.encode('utf-8')).hexdigest()

#     def generate_dna_hash(self, dna_str):
#         """生成 MathML DNA 的指纹"""
#         if not dna_str: return ""
#         return hashlib.md5(dna_str.encode('utf-8')).hexdigest()

#     def get_dual_hash(self, raw_latex, dna_str=""):
#         """
#         供 prepare 脚本调用，一次性返回清洗后的结果和两个哈希
#         """
#         norm_latex = self.clean_latex(raw_latex)
#         return {
#             'norm_latex': norm_latex,
#             'h_latex': self.generate_latex_hash(norm_latex),
#             'h_dna': self.generate_dna_hash(dna_str)
#         }

# # --- 为了兼容之前的 Approach0 索引逻辑，保留以下类 ---
# class Approach0HashIndex:
#     def __init__(self):
#         self.index = {} # key: h_latex, value: list of formula_ids

#     def load(self, path):
#         if Path(path).exists():
#             with open(path, 'rb') as f:
#                 self.index = pickle.load(f)
#         else:
#             print(f"⚠️ 警告：找不到索引文件 {path}")

#     def save(self, path):
#         with open(path, 'wb') as f:
#             pickle.dump(self.index, f)

#     def search(self, h_latex):
#         """根据哈希值秒级返回匹配 ID 列表"""
#         return self.index.get(h_latex, [])


import re
import hashlib
import pickle
from pathlib import Path

# 专家级符号映射表：解决写法异构（如 \| vs ||, ^H vs ^T）
LATEX_SYMBOL_MAPPING = {
    r'\|': '||',
    r'\Vert': '||',
    r'\lbrace': '{',
    r'\rbrace': '}',
    r'\langle': '<',
    r'\rangle': '>',
    r'\varepsilon': r'\epsilon',
    r'\vartheta': r'\theta',
    r'\varkappa': r'\kappa',
    r'\varpi': r'\pi',
    r'\varrho': r'\rho',
    r'\varsigma': r'\sigma',
    r'\varphi': r'\phi',
    r'\le': r'\leq',
    r'\ge': r'\geq',
    r'\ne': r'\neq',
    r'\to': r'\rightarrow',
    r'\gets': r'\leftarrow',
    r'\land': r'\wedge',
    r'\lor': r'\vee',
    r'\lnot': r'\neg',
    r'^H': '^T',
    r'^\dagger': '^T',
    r'^*': '^T',
}

class DualHashGenerator:
    def __init__(self):
        self.font_commands = [
            r'\\mathbf', r'\\mathrm', r'\\mathit', r'\\mathsf', r'\\mathtt', 
            r'\\mathbb', r'\\mathcal', r'\\mathfrak', r'\\text', r'\\bm'
        ]
        self.sorted_symbols = sorted(LATEX_SYMBOL_MAPPING.items(), key=lambda x: len(x[0]), reverse=True)

    def clean_latex(self, latex_str):
        """增强型清洗：返回 (清洗后的字符串, 是否被修改)"""
        if not latex_str: return "", False
        original = latex_str
        
        # 1. 移除定界符
        s = re.sub(r'\$\$?|\\\[|\\\]', '', latex_str)
        # 2. 剥离字体装饰
        for cmd in self.font_commands:
            s = s.replace(cmd, '')
        # 3. 符号别名替换
        for old, new in self.sorted_symbols:
            s = s.replace(old, new)
        # 4. 统一矩阵环境
        s = re.sub(r'\\begin\{(p|b|v|V)matrix\}', r'\\begin{matrix}', s)
        s = re.sub(r'\\end\{(p|b|v|V)matrix\}', r'\\end{matrix}', s)
        # 5. 移除格式装饰符与空格
        s = re.sub(r'\\left|\\right|\\displaystyle|\\limits', '', s)
        s = re.sub(r'\s+', '', s.strip())
        # 6. 简化多余大括号
        s = re.sub(r'\{+([^{}]+)\}+', r'{\1}', s)
        
        # 判定是否发生了增强规范化操作
        base_clean = re.sub(r'\s+', '', re.sub(r'\$\$?|\\\[|\\\]', '', original)).strip()
        was_normalized = (s != base_clean)
        return s, was_normalized

    def generate_latex_hash(self, clean_latex):
        if not clean_latex: return ""
        return hashlib.md5(clean_latex.encode('utf-8')).hexdigest()

class Approach0HashIndex:
    def __init__(self):
        self.index = {} # key: hash, value: list of visual_ids

    def load(self, path):
        if Path(path).exists():
            with open(path, 'rb') as f:
                self.index = pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.index, f)

    def search(self, h_latex):
        return self.index.get(h_latex, [])