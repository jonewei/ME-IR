import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.font_manager as fm

# 尝试查找并使用中文字体
def setup_chinese_font():
    """尝试设置中文字体，如果失败则使用英文"""
    chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 
                     'Noto Sans CJK SC', 'Noto Sans CJK TC',
                     'SimHei', 'Microsoft YaHei', 'STHeiti']
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            rcParams['font.sans-serif'] = [font]
            rcParams['axes.unicode_minus'] = False
            print(f"使用中文字体: {font}")
            return True
    
    print("未找到中文字体，使用英文标签")
    return False

# 设置字体
use_chinese = setup_chinese_font()
rcParams['font.size'] = 11

# 根据是否有中文字体决定标签
if use_chinese:
    labels = {
        'xlabel': '结构权重 ($w_{str}$)',
        'ylabel1': '性能指标',
        'ylabel2': '$p$-value',
        'title': '融合权重敏感性分析与显著性校验 ($k=60$)',
        'optimal_region': '最优区间 [0.2, 0.4]',
        'optimal_point': '最优点\n$w_{str}=0.3$'
    }
else:
    labels = {
        'xlabel': 'Structure Weight ($w_{str}$)',
        'ylabel1': 'Performance Metrics',
        'ylabel2': '$p$-value',
        'title': 'Sensitivity Analysis of Fusion Weight ($k=60$)',
        'optimal_region': 'Optimal Range [0.2, 0.4]',
        'optimal_point': 'Optimal\n$w_{str}=0.3$'
    }

# 数据
weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
P1 = np.array([0.4342, 0.4605, 0.4737, 0.4474, 0.4342, 0.4211, 0.4342, 0.4474, 0.4342, 0.4211])
MRR = np.array([0.4975, 0.5082, 0.5172, 0.5147, 0.5081, 0.5132, 0.5222, 0.5159, 0.5039, 0.4988])
nDCG = np.array([0.2215, 0.2258, 0.2297, 0.2281, 0.2243, 0.2265, 0.2312, 0.2289, 0.2306, 0.2274])
pvalue = np.array([0.0812, 0.0643, 0.0478, 0.0521, 0.1034, 0.1567, 0.2410, 0.3542, 0.4316, 0.5122])
significant = np.array([False, False, True, False, False, False, False, False, False, False])

# 创建图形和双Y轴
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# 绘制最优区间背景
ax1.axvspan(0.2, 0.4, alpha=0.15, color='#0ea5e9', label=labels['optimal_region'])

# 绘制性能指标曲线（左Y轴）
line1 = ax1.plot(weights, P1, 'o-', color='#2563eb', linewidth=2.5, 
                 markersize=6, label='P@1', markeredgecolor='white', markeredgewidth=1.5)
line2 = ax1.plot(weights, MRR, 's-', color='#dc2626', linewidth=2.5, 
                 markersize=6, label='MRR', markeredgecolor='white', markeredgewidth=1.5)
line3 = ax1.plot(weights, nDCG, '^-', color='#16a34a', linewidth=2.5, 
                 markersize=6, label='nDCG@10', markeredgecolor='white', markeredgewidth=1.5)

# 绘制p-value曲线（右Y轴）
line4 = ax2.plot(weights, pvalue, 'd--', color='#9333ea', linewidth=2.5, 
                 markersize=6, label='$p$-value', markeredgecolor='white', markeredgewidth=1.5)

# 绘制显著性阈值线
ax2.axhline(y=0.05, color='#ef4444', linestyle='--', linewidth=2, 
            alpha=0.7, label='$p = 0.05$')

# 标注最优点（w=0.3）
optimal_idx = 2  # w=0.3的索引
ax1.plot(weights[optimal_idx], P1[optimal_idx], 'o', 
         markersize=15, markerfacecolor='none', 
         markeredgecolor='#fbbf24', markeredgewidth=3, zorder=10)
ax1.plot(weights[optimal_idx], MRR[optimal_idx], 's', 
         markersize=15, markerfacecolor='none', 
         markeredgecolor='#fbbf24', markeredgewidth=3, zorder=10)
ax1.plot(weights[optimal_idx], nDCG[optimal_idx], '^', 
         markersize=15, markerfacecolor='none', 
         markeredgecolor='#fbbf24', markeredgewidth=3, zorder=10)

# 添加垂直虚线标注最优点
ax1.axvline(x=0.3, color='#fbbf24', linestyle=':', linewidth=2, alpha=0.6)

# 标注显著性点（添加星标）
for i, sig in enumerate(significant):
    if sig:
        # 在P@1点上方添加星标
        ax1.plot(weights[i], P1[i], '*', color='#fbbf24', 
                markersize=18, zorder=11)

# 设置坐标轴
ax1.set_xlabel(labels['xlabel'], fontsize=13, fontweight='bold')
ax1.set_ylabel(labels['ylabel1'], fontsize=13, fontweight='bold')
ax2.set_ylabel(labels['ylabel2'], fontsize=13, fontweight='bold')

# 设置X轴刻度
ax1.set_xticks(weights)
ax1.set_xlim(0.05, 1.05)

# 设置Y轴范围
ax1.set_ylim(0.20, 0.54)
ax2.set_ylim(0, 0.6)

# 添加网格
ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

# 设置标题
plt.title(labels['title'], fontsize=14, fontweight='bold', pad=20)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper right', frameon=True, shadow=True, 
          fontsize=10, ncol=2)

# 添加文本注释
ax1.text(0.3, 0.52, labels['optimal_point'], 
         ha='center', va='bottom', fontsize=10, 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef3c7', 
                  edgecolor='#fbbf24', linewidth=2),
         fontweight='bold')

# 调整布局
plt.tight_layout()

# 保存为SVG格式
plt.savefig('draw/weight_sensitivity_analysis.svg', format='svg', 
            dpi=300, bbox_inches='tight')
print("图表已保存为: weight_sensitivity_analysis.svg")

# 可选：同时保存PNG格式
plt.savefig('draw/weight_sensitivity_analysis.png', format='png', 
            dpi=300, bbox_inches='tight')
print("图表已保存为: weight_sensitivity_analysis.png")

# 显示图表（如果在本地运行）
# plt.show()

# 我想在论文里写这个内容:
# 本研究初步采用 0.1 作为步幅进行全局扫描,旨在捕捉结构权重对融合性能影响的宏观演变趋势。实验结果显示,性能曲线在 $[0.2, 0.4]$ 区间表现出良好的平滑性与统计显著性。由于该区间已展现出稳定的最优特征,且 RRF 算法对微小权重偏移具有较好的鲁棒性,因此本研究认为 0.1 的粒度已足以支撑对‘拓扑锚定’效应的定性与定量分析。
# 能不能帮我把上面的数据用python生成一个漂亮的图、