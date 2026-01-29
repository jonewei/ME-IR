import torch
from sentence_transformers import CrossEncoder

model = CrossEncoder("artifacts/cross_encoder_model", device="cuda")

# 找一个你确定的正样本对（从数据集中找一个 label=1 的）
query = "||A||_2=\\sqrt{\\rho(A^TA)}"
positive = "||A||_2=\\sqrt{\\rho(A^TA)}" # 完全一样
negative = "a^2 + b^2 = c^2" # 完全无关

pairs = [[query, positive], [query, negative]]
scores = model.predict(pairs)

print(f"📊 正样本得分: {scores[0]}")
print(f"📊 负样本得分: {scores[1]}")

if scores[0] < scores[1]:
    print("❌ 结论：模型学反了！它认为不相关的公式更相似。")
else:
    print("✅ 结论：得分逻辑正常，可能是其他问题。")