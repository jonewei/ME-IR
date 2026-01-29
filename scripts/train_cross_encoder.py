import json
import os
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, CrossEncoder
import math

# 1. 配置参数 (建议使用绝对路径)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data/train_cross_encoder.jsonl")
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "artifacts/cross_encoder_model")
# BASE_MODEL = "bert-base-uncased" 
# BATCH_SIZE = 64 # 4090 显存大，直接上 64 提速
# NUM_EPOCHS = 3
# LEARNING_RATE = 2e-5
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2" # 换成精排专用模型
NUM_EPOCHS = 10 # 增加轮数
LEARNING_RATE = 5e-6 # 降低学习率
BATCH_SIZE = 64 # 4090 显存大，直接上 64 提速

def extract_latex(item):
    if isinstance(item, str): return item
    if isinstance(item, dict): return item.get("latex_norm") or item.get("latex") or ""
    return str(item)

def train():
    # --- 关键修复：确保目录存在 ---
    if not os.path.exists(MODEL_OUTPUT_PATH):
        os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
        print(f"📁 已创建输出目录: {MODEL_OUTPUT_PATH}")

    print(f"🚀 重新启动训练 (基于 {BASE_MODEL})...")
    
    # 加载数据逻辑 (保持不变)
    train_examples = []
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            q = extract_latex(data['texts'][0])
            d = extract_latex(data['texts'][1])
            train_examples.append(InputExample(texts=[q, d], label=float(data['label'])))
    
    print(f"📦 有效数据量: {len(train_examples)}")

    model = CrossEncoder(BASE_MODEL, num_labels=1, device="cuda")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1)

    # 开始训练
    model.fit(
        train_dataloader=train_dataloader,
        epochs=NUM_EPOCHS,
        optimizer_params={'lr': LEARNING_RATE},
        warmup_steps=warmup_steps,
        output_path=MODEL_OUTPUT_PATH, # 虽然 fit 会存，但有时会因异常跳过
        show_progress_bar=True
    )

    # --- 关键修复：显式强制保存 ---
    print("💾 正在执行显式保存...")
    model.save(MODEL_OUTPUT_PATH)
    model.tokenizer.save_pretrained(MODEL_OUTPUT_PATH)
    
    # 检查是否真的存上了
    if os.path.exists(os.path.join(MODEL_OUTPUT_PATH, "config.json")):
        print(f"✅ 验证成功！模型已落地: {MODEL_OUTPUT_PATH}")
    else:
        print("❌ 警告：保存动作执行了，但 config.json 仍不存在，请检查磁盘空间！")

if __name__ == "__main__":
    train()