#!/usr/bin/env python3
"""
Kaggle Toxic Comment Classification Challenge
使用真实数据的完整数据分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("📊 KAGGLE TOXIC COMMENT CLASSIFICATION - 真实数据分析")
print("="*70)

# ==================== 1. 加载真实数据 ====================
print("\n📥 加载真实数据...")
train_df = pd.read_csv('data/train.csv')

print(f"✅ 数据加载完成!")
print(f"   总样本数: {len(train_df):,}")
print(f"   列: {list(train_df.columns)}")
print(f"\n前3条数据:")
print(train_df.head(3))

# ==================== 2. 数据分析 (Task 1a) ====================
print("\n" + "="*70)
print("📊 2. 详细数据分析")
print("="*70)

toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# 2.1 类别分布
print("\n2.1 各类别样本数量和比例:")
print("-" * 60)
class_stats = []
for col in toxicity_cols:
    count = train_df[col].sum()
    percentage = count / len(train_df) * 100
    class_stats.append({
        '类别': col,
        '样本数': count,
        '比例(%)': f"{percentage:.2f}%"
    })
    print(f"  {col:20}: {count:6,} ({percentage:6.2f}%)")

# 检查多标签情况
print("\n2.2 多标签分析:")
print("-" * 60)
train_df['total_toxic'] = train_df[toxicity_cols].sum(axis=1)
print(f"  无任何毒性: {(train_df['total_toxic'] == 0).sum():,} ({(train_df['total_toxic'] == 0).mean()*100:.2f}%)")
print(f"  单一毒性: {(train_df['total_toxic'] == 1).sum():,}")
print(f"  多种毒性: {(train_df['total_toxic'] > 1).sum():,}")
print(f"  最多毒性标签数: {train_df['total_toxic'].max()}")

# 可视化类别分布
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 图1: 各类别样本数
ax1 = axes[0, 0]
counts = [train_df[col].sum() for col in toxicity_cols]
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(toxicity_cols)))
bars = ax1.bar(range(len(toxicity_cols)), counts, color=colors)
ax1.set_xticks(range(len(toxicity_cols)))
ax1.set_xticklabels(toxicity_cols, rotation=45, ha='right')
ax1.set_ylabel('样本数')
ax1.set_title('各类别毒性评论数量', fontsize=14, fontweight='bold')
for i, (bar, count) in enumerate(zip(bars, counts)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
             f'{count:,}', ha='center', va='bottom', fontsize=9)

# 图2: 多标签分布
ax2 = axes[0, 1]
toxic_counts = train_df['total_toxic'].value_counts().sort_index()
ax2.bar(toxic_counts.index, toxic_counts.values, color='coral')
ax2.set_xlabel('毒性标签数量')
ax2.set_ylabel('评论数')
ax2.set_title('每篇评论的毒性标签数量分布', fontsize=14, fontweight='bold')

# 图3: 类别间的相关性
ax3 = axes[1, 0]
corr = train_df[toxicity_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax3, 
            fmt='.2f', square=True)
ax3.set_title('各类别间的相关性', fontsize=14, fontweight='bold')

# 图4: 饼图
ax4 = axes[1, 1]
counts_pie = [train_df[col].sum() for col in toxicity_cols]
ax4.pie(counts_pie, labels=toxicity_cols, autopct='%1.1f%%', startangle=90)
ax4.set_title('各类别占比', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('real_class_distribution.png', dpi=150, bbox_inches='tight')
print("\n📊 类别分布图已保存: real_class_distribution.png")

# 2.3 文本长度分析
print("\n2.3 文本长度分析:")
print("-" * 60)
train_df['text_length'] = train_df['comment_text'].str.len()
train_df['word_count'] = train_df['comment_text'].str.split().str.len()

print(f"  平均字符数: {train_df['text_length'].mean():.2f}")
print(f"  中位数字符数: {train_df['text_length'].median():.2f}")
print(f"  最大字符数: {train_df['text_length'].max():,}")
print(f"  平均词数: {train_df['word_count'].mean():.2f}")

# 按毒性分析文本长度
print("\n  按毒性分类的文本长度:")
for col in ['toxic', 'obscene', 'insult']:
    toxic_mean = train_df[train_df[col] == 1]['text_length'].mean()
    normal_mean = train_df[train_df[col] == 0]['text_length'].mean()
    print(f"    {col:15}: 有毒={toxic_mean:7.1f}, 正常={normal_mean:7.1f}")

# 可视化文本长度
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax1 = axes[0]
normal_lengths = train_df[train_df['toxic'] == 0]['text_length']
toxic_lengths = train_df[train_df['toxic'] == 1]['text_length']
ax1.hist(normal_lengths, bins=50, alpha=0.7, label='正常', density=True, range=(0, 1000))
ax1.hist(toxic_lengths, bins=50, alpha=0.7, label='有毒', density=True, range=(0, 1000))
ax1.set_xlabel('字符数')
ax1.set_ylabel('密度')
ax1.set_title('文本长度分布 (正常 vs 有毒)', fontsize=14, fontweight='bold')
ax1.legend()

ax2 = axes[1]
ax2.boxplot([normal_lengths[normal_lengths < 2000], 
             toxic_lengths[toxic_lengths < 2000]], 
            labels=['正常', '有毒'])
ax2.set_ylabel('字符数')
ax2.set_title('文本长度箱线图', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('real_text_length.png', dpi=150, bbox_inches='tight')
print("\n📊 文本长度分布图已保存: real_text_length.png")

# 2.4 最常见词汇分析
print("\n2.4 各类别的最常见词汇:")
print("-" * 60)

def get_words(texts):
    """提取所有单词"""
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
        all_words.extend(words)
    return all_words

# 停用词
stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'dad', 'mom'}

# 分析每个类别的常见词
for col in ['toxic', 'obscene', 'insult']:
    toxic_texts = train_df[train_df[col] == 1]['comment_text'].sample(min(5000, train_df[col].sum()), random_state=42)
    normal_texts = train_df[train_df[col] == 0]['comment_text'].sample(5000, random_state=42)
    
    toxic_words = Counter([w for w in get_words(toxic_texts) if w not in stop_words])
    normal_words = Counter([w for w in get_words(normal_texts) if w not in stop_words])
    
    print(f"\n{col.upper()}类别:")
    print(f"  有毒评论常见词: {[w for w, c in toxic_words.most_common(8)]}")
    print(f"  正常评论常见词: {[w for w, c in normal_words.most_common(8)]}")

# 生成词云
print("\n📊 生成词云...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 有毒评论词云
toxic_text = ' '.join(train_df[train_df['toxic'] == 1]['comment_text'].sample(5000, random_state=42))
wordcloud_toxic = WordCloud(width=800, height=400, background_color='white', 
                             max_words=100, colormap='Reds').generate(toxic_text)
axes[0].imshow(wordcloud_toxic, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('有毒评论词云', fontsize=16, fontweight='bold')

# 正常评论词云
normal_text = ' '.join(train_df[train_df['toxic'] == 0]['comment_text'].sample(5000, random_state=42))
wordcloud_normal = WordCloud(width=800, height=400, background_color='white', 
                              max_words=100, colormap='Blues').generate(normal_text)
axes[1].imshow(wordcloud_normal, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('正常评论词云', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('real_wordcloud.png', dpi=150, bbox_inches='tight')
print("📊 词云图已保存: real_wordcloud.png")

print("\n" + "="*70)
print("✅ 数据分析完成!")
print("="*70)

# 保存统计数据
stats = {
    '总样本数': len(train_df),
    '类别分布': {col: int(train_df[col].sum()) for col in toxicity_cols},
    '平均文本长度': float(train_df['text_length'].mean()),
    '平均词数': float(train_df['word_count'].mean())
}

import json
with open('data_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n📁 统计数据已保存: data_stats.json")
