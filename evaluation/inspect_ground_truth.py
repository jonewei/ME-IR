import json
import csv
from pathlib import Path

def inspect_gt():
    # 1. 加载你当前的标注
    with open("data/processed/relevance_labels.json", 'r') as f:
        relevance = json.load(f)
    
    # 获取 B.301 的所有标准答案 ID
    target_topic = "B.301"
    gt_ids = list(relevance.get(target_topic, {}).keys())
    print(f"📊 Topic {target_topic} 在标注中有 {len(gt_ids)} 个相关公式 ID。")
    print(f"🔎 正在语料库中寻找这些 ID 的实际内容...")

    # 2. 从原始分片中寻找这些 ID
    # 我们随便找前 5 个分片看看
    latex_dir = Path("data/arqmath3/latex_representation_v3")
    found_count = 0
    
    for f in sorted(latex_dir.glob("*.tsv"))[:10]: # 先看 10 个分片
        with open(f, 'r', encoding='utf-8') as fin:
            reader = csv.reader(fin, delimiter='\t')
            next(reader)
            for row in reader:
                fid = row[0].strip()
                if fid in gt_ids:
                    print(f"✅ 找到匹配 ID: {fid}")
                    print(f"   内容: {row[8]}")
                    found_count += 1
    
    if found_count == 0:
        print("\n❌ 警报：在语料库的前 10 个分片中，完全找不到标注文件里的 ID！")
        print("💡 结论：你的标注文件 (relevance_labels.json) 使用的 ID 类型与语料库 (TSV) 不一致。")

if __name__ == "__main__":
    inspect_gt()