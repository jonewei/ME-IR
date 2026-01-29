import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, FancyBboxPatch
from collections import defaultdict

# 设置学术风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

def analyze_noise_wall():
    """分析并可视化 ARQMath-3 的真实语义噪声墙"""
    
    # ========== 加载数据 ==========
    print("📂 加载数据...")
    with open("data/qrel_76_expert.json", "r") as f:
        qrel = json.load(f)
    
    with open("results/raw_sem_scores.json", "r") as f:
        sem_scores = json.load(f)
    
    # ========== 统计分析 ==========
    print("📊 统计分析中...")
    
    stats = {
        'avg_top1_sim': [],
        'avg_top10_sim': [],
        'avg_top100_sim': [],
        'truth_avg_rank': [],
        'truth_avg_sim': [],
        'noise_density': []
    }
    
    for qid in qrel.keys():
        if qid not in sem_scores:
            continue
        
        # 获取真值 ID
        truth_ids = set(qrel[qid].keys())
        
        # 获取候选分数（已排序）
        candidates = sem_scores[qid]
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        # Top-K 平均相似度
        stats['avg_top1_sim'].append(sorted_candidates[0][1])
        stats['avg_top10_sim'].append(np.mean([s for _, s in sorted_candidates[:10]]))
        stats['avg_top100_sim'].append(np.mean([s for _, s in sorted_candidates[:100]]))
        
        # 真值的排名和分数
        truth_ranks = []
        truth_sims = []
        for rank, (fid, sim) in enumerate(sorted_candidates, 1):
            if fid in truth_ids:
                truth_ranks.append(rank)
                truth_sims.append(sim)
        
        if truth_ranks:
            stats['truth_avg_rank'].append(np.mean(truth_ranks))
            stats['truth_avg_sim'].append(np.mean(truth_sims))
        
        # 噪声密度（相似度 > 0.94 的比例）
        high_sim_count = sum(1 for _, s in sorted_candidates[:100] if s > 0.94)
        stats['noise_density'].append(high_sim_count / 100)
    
    # ========== 选择最具代表性的查询 ==========
    # 选择真值排名最靠后的查询（噪声墙最严重）
    worst_qid = None
    worst_rank = 0
    
    for qid in qrel.keys():
        if qid not in sem_scores:
            continue
        
        truth_ids = set(qrel[qid].keys())
        candidates = sem_scores[qid]
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (fid, sim) in enumerate(sorted_candidates, 1):
            if fid in truth_ids:
                if rank > worst_rank:
                    worst_rank = rank
                    worst_qid = qid
                break
    
    print(f"\n🎯 选择最具代表性的查询: {worst_qid}")
    print(f"   真值最差排名: #{worst_rank}")
    
    # ========== 绘制可视化 ==========
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # --- 子图 1: 单查询噪声墙 ---
    ax1 = fig.add_subplot(gs[0, :])
    draw_single_query_noise_wall(ax1, worst_qid, qrel, sem_scores)
    
    # --- 子图 2: 全局噪声密度分布 ---
    ax2 = fig.add_subplot(gs[1, 0])
    draw_noise_density_distribution(ax2, stats)
    
    # --- 子图 3: 真值淹没统计 ---
    ax3 = fig.add_subplot(gs[1, 1])
    draw_truth_burial_stats(ax3, stats)
    
    # ========== 保存 ==========
    plt.savefig("draw/ARQMath3_Real_Noise_Wall_Analysis.svg", 
                format='svg', bbox_inches='tight', dpi=300)
    plt.savefig("draw/ARQMath3_Real_Noise_Wall_Analysis.png", 
                format='png', bbox_inches='tight', dpi=300)
    
    print("\n✅ 可视化已保存:")
    print("   📄 SVG: draw/ARQMath3_Real_Noise_Wall_Analysis.svg")
    print("   🖼️  PNG: draw/ARQMath3_Real_Noise_Wall_Analysis.png")
    
    # ========== 打印统计摘要 ==========
    print("\n" + "="*60)
    print("📈 ARQMath-3 语义噪声墙统计摘要")
    print("="*60)
    print(f"平均 Top-1 相似度:   {np.mean(stats['avg_top1_sim']):.4f}")
    print(f"平均 Top-10 相似度:  {np.mean(stats['avg_top10_sim']):.4f}")
    print(f"平均 Top-100 相似度: {np.mean(stats['avg_top100_sim']):.4f}")
    print(f"真值平均排名:        #{np.mean(stats['truth_avg_rank']):.1f}")
    print(f"真值平均相似度:      {np.mean(stats['truth_avg_sim']):.4f}")
    print(f"噪声密度 (Sim>0.94): {np.mean(stats['noise_density'])*100:.1f}%")
    print("="*60)
    
    plt.show()


def draw_single_query_noise_wall(ax, qid, qrel, sem_scores):
    """绘制单个查询的噪声墙"""
    
    truth_ids = set(qrel[qid].keys())
    candidates = sem_scores[qid]
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    
    # 取 Top-500
    top500 = sorted_candidates[:500]
    ranks = np.arange(1, 501)
    sims = [s for _, s in top500]
    
    # 标记真值位置
    truth_ranks = []
    truth_sims = []
    for rank, (fid, sim) in enumerate(top500, 1):
        if fid in truth_ids:
            truth_ranks.append(rank)
            truth_sims.append(sim)
    
    # 绘制噪声墙
    ax.fill_between(ranks, sims, alpha=0.3, color='red', label='Noise Wall')
    ax.plot(ranks, sims, color='darkred', linewidth=2, alpha=0.7)
    
    # 标记真值
    if truth_ranks:
        ax.scatter(truth_ranks, truth_sims, 
                  c='gold', s=200, marker='*', 
                  edgecolors='black', linewidths=2,
                  label=f'Ground Truth (n={len(truth_ranks)})',
                  zorder=10)
    
    # 阈值线
    ax.axhline(y=0.94, color='orange', linestyle='--', 
               linewidth=2, alpha=0.8, label='High Similarity Threshold (0.94)')
    
    ax.set_title(f'A. Semantic Noise Wall for Query {qid}\n'
                 f'Top-500 Candidates | Avg Sim: {np.mean(sims):.4f}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.4)


def draw_noise_density_distribution(ax, stats):
    """绘制噪声密度分布"""
    
    densities = np.array(stats['noise_density']) * 100
    
    ax.hist(densities, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(densities), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(densities):.1f}%')
    
    ax.set_title('B. Noise Density Distribution\n(Sim > 0.94 in Top-100)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Noise Density (%)', fontsize=11)
    ax.set_ylabel('Number of Queries', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.4)


def draw_truth_burial_stats(ax, stats):
    """绘制真值淹没统计"""
    
    ranks = stats['truth_avg_rank']
    sims = stats['truth_avg_sim']
    
    scatter = ax.scatter(ranks, sims, c=ranks, cmap='RdYlGn_r', 
                        s=100, alpha=0.6, edgecolors='black')
    
    ax.axhline(y=np.mean(stats['avg_top10_sim']), 
               color='blue', linestyle='--', alpha=0.5,
               label='Avg Top-10 Similarity')
    
    ax.set_title('C. Ground Truth Burial Analysis\n'
                 'Rank vs Similarity',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Average Rank of Ground Truth', fontsize=11)
    ax.set_ylabel('Average Similarity', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Rank (worse →)', fontsize=10)


if __name__ == "__main__":
    analyze_noise_wall()
