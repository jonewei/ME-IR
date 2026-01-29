# """
# 使用 latex2mathml 的简化版本
# """

# import re
# import logging
# from typing import Optional

# logger = logging.getLogger(__name__)

# try:
#     from latex2mathml.converter import convert as latex2mathml_convert
#     HAS_LATEX2MATHML = True
# except ImportError:
#     HAS_LATEX2MATHML = False
#     logger.warning("latex2mathml not installed. Run: pip install latex2mathml")


# def latex_to_mathml(latex_str: str) -> Optional[str]:
#     """
#     将 LaTeX 转换为 MathML 骨架
    
#     Args:
#         latex_str: LaTeX 字符串
        
#     Returns:
#         MathML 标签序列（逗号分隔），失败返回 None
#     """
#     if not HAS_LATEX2MATHML:
#         logger.error("latex2mathml not available")
#         return None
    
#     if not latex_str or not latex_str.strip():
#         return None
    
#     # 1. 预处理 LaTeX
#     latex_str = preprocess_latex(latex_str)
    
#     try:
#         # 2. 转换为 MathML
#         mathml_xml = latex2mathml_convert(latex_str)
        
#         # 3. 提取标签骨架
#         skel = extract_skeleton(mathml_xml)
        
#         return skel
        
#     except Exception as e:
#         logger.debug(f"Conversion failed for: {latex_str[:50]}... Error: {e}")
#         return None


# def preprocess_latex(latex_str: str) -> str:
#     """
#     预处理 LaTeX：移除环境标签和多余空格
#     """
#     # 移除 align, equation 等环境
#     latex_str = re.sub(r'\\begin\{[^}]+\}', '', latex_str)
#     latex_str = re.sub(r'\\end\{[^}]+\}', '', latex_str)
    
#     # 移除多余空格
#     latex_str = re.sub(r'\s+', ' ', latex_str).strip()
    
#     # 移除 & 和 \\ (对齐符号)
#     latex_str = latex_str.replace('&', '').replace('\\\\', '')
    
#     return latex_str


# def extract_skeleton(mathml_xml: str) -> str:
#     """
#     从 MathML XML 中提取标签骨架
#     """
#     # 提取所有开始标签
#     tags = re.findall(r'<([a-z]+)', mathml_xml.lower())
    
#     # 过滤冗余标签
#     ignored_tags = {
#         'math', 'semantics', 'annotation', 'annotation-xml',
#         'mstyle', 'mrow', 'mtext'
#     }
    
#     filtered_tags = [t for t in tags if t not in ignored_tags]
    
#     return ','.join(filtered_tags)


# # ========== 测试 ==========

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     test_cases = [
#         r"\frac{1}{\sqrt{n}}",
#         r"x^2 + y^2 = z^2",
#         r"\int_0^1 f(x) dx",
#         r"\sum_{i=1}^n i",
#         r"\begin{align*} a &= b \\ c &= d \end{align*}",
#     ]
    
#     print("=" * 60)
#     print("LaTeX → MathML 骨架转换测试")
#     print("=" * 60)
    
#     for i, latex in enumerate(test_cases, 1):
#         print(f"\n{i}. LaTeX: {latex}")
#         skel = latex_to_mathml(latex)
#         print(f"   Skeleton: {skel}")
    
#     print("\n" + "=" * 60)
"""
LaTeX 归一化工具 - SymPy 版本

使用 SymPy 进行深度归一化，处理数学等价性
"""

import re
import logging
import hashlib
from typing import Optional

logger = logging.getLogger(__name__)

# 尝试导入 SymPy
try:
    from latex2sympy2 import latex2sympy
    from sympy import latex, simplify
    SYMPY_AVAILABLE = True
    logger.info("✅ SymPy loaded successfully")
except ImportError as e:
    SYMPY_AVAILABLE = False
    logger.warning(f"⚠️  SymPy not available: {e}")
    logger.warning("   Install with: pip install sympy latex2sympy2")

##############
import signal
from functools import lru_cache

# 添加超时装饰器
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

@lru_cache(maxsize=100000)  # 缓存 10 万个结果
def normalize_with_sympy(latex_str: str, timeout_seconds: float = 0.5) -> Optional[str]:
    """
    使用 SymPy 进行深度归一化（带超时和缓存）
    
    Args:
        latex_str: 原始 LaTeX 字符串
        timeout_seconds: 超时时间（秒）
        
    Returns:
        归一化后的 LaTeX，失败返回 None
    """
    if not SYMPY_AVAILABLE:
        return None
    
    if not latex_str or not latex_str.strip():
        return None
    
    # 预处理
    latex_str = preprocess_latex(latex_str)
    
    # 设置超时（仅在 Linux 上）
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
    
    try:
        expr = latex2sympy(latex_str)
        expr = simplify(expr)
        normalized = latex(expr)
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # 取消超时
        
        return normalized
        
    except TimeoutException:
        logger.debug(f"SymPy timeout for: {latex_str[:50]}...")
        return None
    except Exception as e:
        logger.debug(f"SymPy parsing failed for: {latex_str[:50]}... Error: {e}")
        return None
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

##############





def preprocess_latex(latex_str: str) -> str:
    """
    预处理 LaTeX 字符串
    
    移除环境标签和多余符号，为 SymPy 解析做准备
    """
    if not latex_str:
        return ""
    
    # 移除 align, equation 等环境
    latex_str = re.sub(r'\\begin\{align\*?\}', '', latex_str)
    latex_str = re.sub(r'\\end\{align\*?\}', '', latex_str)
    latex_str = re.sub(r'\\begin\{equation\*?\}', '', latex_str)
    latex_str = re.sub(r'\\end\{equation\*?\}', '', latex_str)
    latex_str = re.sub(r'\\begin\{cases\}', '', latex_str)
    latex_str = re.sub(r'\\end\{cases\}', '', latex_str)
    
    # 移除换行符和对齐符
    latex_str = latex_str.replace('\\\\', ' ')
    latex_str = latex_str.replace('&', '')
    
    # 移除多余的修饰命令
    latex_str = re.sub(r'\\limits', '', latex_str)
    latex_str = re.sub(r'\\displaystyle', '', latex_str)
    
    # 移除多余空格
    latex_str = re.sub(r'\s+', ' ', latex_str).strip()
    
    return latex_str


# def normalize_with_sympy(latex_str: str) -> Optional[str]:
#     """
#     使用 SymPy 进行深度归一化
    
#     Args:
#         latex_str: 原始 LaTeX 字符串
        
#     Returns:
#         归一化后的 LaTeX，失败返回 None
#     """
#     if not SYMPY_AVAILABLE:
#         return None
    
#     if not latex_str or not latex_str.strip():
#         return None
    
#     # 预处理
#     latex_str = preprocess_latex(latex_str)
    
#     try:
#         # 解析为 SymPy 表达式
#         expr = latex2sympy(latex_str)
        
#         # 简化表达式
#         expr = simplify(expr)
        
#         # 重新生成标准 LaTeX
#         normalized = latex(expr)
        
#         return normalized
        
#     except Exception as e:
#         logger.debug(f"SymPy parsing failed for: {latex_str[:50]}... Error: {e}")
#         return None


def basic_normalize(latex_str: str) -> str:
    """
    基础归一化（SymPy 失败时的回退方案）
    """
    # 预处理
    latex_str = preprocess_latex(latex_str)
    
    # 移除所有空格
    latex_str = re.sub(r'\s+', '', latex_str)
    
    # 统一符号变体
    replacements = {
        r'\parallel': r'\|',
        '||': r'\|',
        r'\leq': r'\le',
        r'\geq': r'\ge',
        r'\infty': r'\infty',
        r'\left': '',
        r'\right': '',
        r'\cdot': '*',
        r'\times': '*',
    }
    
    for old, new in replacements.items():
        latex_str = latex_str.replace(old, new)
    
    return latex_str


def normalize_latex_for_matching(latex_str: str) -> str:
    """
    智能归一化：优先使用 SymPy，失败则回退到基础方法
    
    Args:
        latex_str: 原始 LaTeX 字符串
        
    Returns:
        归一化后的字符串（用于匹配）
    """
    if not latex_str:
        return ""
    
    # 优先使用 SymPy
    sympy_result = normalize_with_sympy(latex_str)
    if sympy_result:
        return sympy_result
    
    # 回退到基础方法
    return basic_normalize(latex_str)


def latex_hash(latex_str: str) -> str:
    """
    基于归一化的哈希
    
    Args:
        latex_str: LaTeX 字符串
        
    Returns:
        MD5 哈希值
    """
    normalized = normalize_latex_for_matching(latex_str)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


# ========== 测试函数 ==========

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("🔧 SymPy LaTeX 归一化测试")
    print("=" * 80)
    
    # 测试用例
    test_cases = [
        (r"\frac{1}{\sqrt{n}}", "标准格式"),
        (r"\frac {1} {\sqrt {n}}", "带空格"),
        (r"1/\sqrt{n}", "不同写法"),
        (r"\begin{align*} \frac{1}{\sqrt{n}} \end{align*}", "带环境"),
        (r"x^2 + y^2 = z^2", "简单公式"),
        (r"\int_0^1 f(x) dx", "积分"),
        (r"\sum_{i=1}^n i", "求和"),
    ]
    
    print(f"\n{'原始 LaTeX':<50} | {'归一化结果':<40} | 哈希值")
    print("-" * 130)
    
    for latex, desc in test_cases:
        normalized = normalize_latex_for_matching(latex)
        hash_val = latex_hash(latex)
        
        # 截断显示
        latex_short = (latex[:47] + '...') if len(latex) > 50 else latex
        norm_short = (normalized[:37] + '...') if len(normalized) > 40 else normalized
        
        print(f"{latex_short:<50} | {norm_short:<40} | {hash_val[:16]}...")
    
    print("=" * 80)
    
    # 测试等价性
    print("\n🔍 等价性测试:")
    print("以下公式在数学上等价，应该生成相同或相似的归一化结果：")
    
    variants = [
        r"\frac{1}{\sqrt{n}}",
        r"\frac {1} {\sqrt {n}}",
        r"1/\sqrt{n}",
    ]
    
    results = [(v, normalize_latex_for_matching(v), latex_hash(v)) for v in variants]
    
    for i, (latex, norm, h) in enumerate(results, 1):
        print(f"{i}. {latex:<30} → {norm:<30} | {h[:16]}...")
    
    # 检查哈希一致性
    hashes = [r[2] for r in results]
    if len(set(hashes)) == 1:
        print("\n✅ 所有变体生成相同哈希（完美！）")
    elif len(set(hashes)) == len(hashes):
        print(f"\n⚠️  所有哈希都不同（归一化可能不够强）")
    else:
        print(f"\n🟡 部分哈希相同（部分归一化成功）")
    
    print("=" * 80)

