import json
import re
import math
import pickle
import os
from collections import defaultdict, Counter
from tqdm import tqdm

class PathInvertedIndex:
    def __init__(self, path_length=2):
        self.path_length = path_length
        self.index = defaultdict(list)  # Key: Path string, Value: List of Formula IDs
        self.formula_lengths = {}      # 用于长度归一化
        self.idf = {}                  # 存储路径权重
        self.total_formulas = 0

    def _extract_latex(self, item):
        """兼容字符串和嵌套字典的提取逻辑"""
        if isinstance(item, str): return item
        if isinstance(item, dict):
            return item.get("latex_norm") or item.get("latex") or ""
        return str(item) if item is not None else ""

    def _extract_paths(self, latex):
        """核心解析：将 LaTeX 拆解为符号路径"""
        # 移除空格，保持转义符
        latex = re.sub(r'\s+', '', self._extract_latex(latex))
        # 符号化拆解：匹配命令(\sum)、括号、数字、变量及算子
        tokens = re.findall(r'\\[a-zA-Z]+|[{}]|[0-9a-zA-Z]|[\+\-\*/=\(\)_^]', latex)
        
        # 提取 N-gram 结构路径
        paths = []
        for i in range(len(tokens) - self.path_length + 1):
            path = "->".join(tokens[i : i + self.path_length])
            paths.append(path)
        return paths

    def build_index(self, formulas_dict):
        """构建大规模倒排索引 (TF-IDF 模式)"""
        print(f"🏗️ 正在构建子结构索引 (L={self.path_length})...")
        self.total_formulas = len(formulas_dict)
        df_counter = Counter()

        for fid, data in tqdm(formulas_dict.items()):
            paths = self._extract_paths(data)
            if not paths: continue
            
            self.formula_lengths[fid] = len(paths)
            unique_paths = set(paths)
            
            for p in unique_paths:
                self.index[p].append(fid)
                df_counter[p] += 1
        
        # 计算 IDF 权重 (log 缩放)
        print("📊 计算路径全局权重 (IDF)...")
        for path, df in df_counter.items():
            self.idf[path] = math.log10(self.total_formulas / (df + 1))
        print(f"✅ 倒排索引构建完成，唯一路径数：{len(self.index)}")

    def search(self, query_latex, top_k=1000):
        """执行路径匹配检索"""
        q_paths = self._extract_paths(query_latex)
        if not q_paths: return []

        scores = defaultdict(float)
        q_path_counts = Counter(q_paths)

        # 命中路径打分累加
        for path, q_count in q_path_counts.items():
            if path in self.index:
                weight = self.idf.get(path, 1.0)
                for fid in self.index[path]:
                    # TF-IDF 基础得分
                    scores[fid] += (q_count * weight)

        # 长度归一化（防止长公式在结构匹配中获得不公平的高分）
        for fid in scores:
            scores[fid] /= (self.formula_lengths.get(fid, 1) ** 0.5)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"💾 索引已保存至: {file_path}")

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)