import json
import os
import sys

# 确保能找到项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.path_inverted_index import PathInvertedIndex



def main():
    CORPUS_PATH = "data/processed/formulas.json"
    INDEX_SAVE_PATH = "artifacts/substructure_index.pkl"

    if not os.path.exists(CORPUS_PATH):
        print(f"❌ 找不到公式库文件: {CORPUS_PATH}")
        return

    print("📖 正在加载全量公式库 (8.41M)...")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        formulas = json.load(f)

    # 初始化索引系统
    # path_length=2 适合大多数“包含关系”匹配，若追求更极端精度可设为 3
    index_system = PathInvertedIndex(path_length=2)
    
    # 构建并保存
    index_system.build_index(formulas)
    index_system.save(INDEX_SAVE_PATH)

    print("✨ 子结构索引构建任务圆满完成！")

if __name__ == "__main__":
    main()