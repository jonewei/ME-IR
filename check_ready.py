import json, sys

# 1. 检查 coverage
with open("data/processed/stats.json") as f:
    st = json.load(f)
cov = st.get("coverage_rate", 0)
print("coverage_rate =", cov)
if cov < 0.85:
    sys.exit("❌ coverage < 85 %")

# 2. 检查 relevance
with open("data/arqmath3/relevance_labels.json") as f:
    qrel = json.load(f)
if not any(v == 1 for vs in qrel.values() for v in vs.values()):
    sys.exit("❌ relevance_labels 没有正例")
print("✅ 检查通过，可以跑评测")