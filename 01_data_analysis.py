#!/usr/bin/env python3
"""
Kaggle Toxic Comment Classification Challenge
完整数据分析和机器学习解决方案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("Kaggle Toxic Comment Classification Challenge")
print("数据分析和机器学习解决方案")
print("="*60)

# ==================== 1. 数据加载 ====================
print("\n📥 1. 数据加载...")

# 由于无法下载真实数据，创建模拟数据进行演示
# 实际使用时替换为: train_df = pd.read_csv('data/train.csv')

np.random.seed(42)
n_samples = 5000

# 创建模拟数据
train_df = pd.DataFrame({
    'id': range(n_samples),
    'comment_text': [
        'This is a normal comment ' + str(i) if np.random.random() > 0.2
        else 'You are stupid and idiot ' + str(i) if np.random.random() > 0.5
        else 'I hate you so much ' + str(i) if np.random.random() > 0.5
        else 'This is great ' + str(i)
        for i in range(n_samples)
    ],
    'toxic': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'severe_toxic': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
    'obscene': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    'threat': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
    'insult': np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
    'identity_hate': np.random.choice([0, 1], n_samples, p=[0.97, 0.03])
})

print(f"✅ 数据加载完成: {len(train_df)} 条评论")
print(f"\n前5条数据:")
print(train_df.head())

# ==================== 2. 数据分析 (Task 1a) ====================
print("\n" + "="*60)
print("📊 2. 数据分析 (Task 1a)")
print("="*60)

# 2.1 类别分布
toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
class_distribution = train_df[toxicity_cols].sum().sort_values(ascending=False)

print("\n2.1 各类别样本数量:")
for col in toxicity_cols:
    count = train_df[col].sum()
    percentage = count / len(train_df) * 100
    print(f"  {col:20}: {count:5} ({percentage:.2f}%)")

# 检查是否不平衡
print("\n⚖️  数据集平衡性分析:")
for col in toxicity_cols:
    count = train_df[col].sum()
    if count / len(train_df) < 0.05:
        print(f"  ⚠️  {col}: 严重不平衡 (正样本 < 5%)")
    elif count / len(train_df) < 0.2:
        print(f"  ⚠️  {col}: 轻度不平衡")
    else:
        print(f"  ✅ {col}: 相对平衡")

# 可视化类别分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 条形图
class_distribution.plot(kind='bar', ax=ax1, color='coral')
ax1.set_title('Toxicity Class Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Toxicity Type')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# 饼图
ax2.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('Percentage Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
print("\n📊 类别分布图已保存: class_distribution.png")

# 2.2 文本长度分析
train_df['text_length'] = train_df['comment_text'].apply(len)
train_df['word_count'] = train_df['comment_text'].apply(lambda x: len(x.split()))

print("\n2.2 文本长度统计:")
print(f"  平均字符数: {train_df['text_length'].mean():.2f}")
print(f"  平均词数: {train_df['word_count'].mean():.2f}")

# 按类别分析文本长度
print("\n按类别的平均文本长度:")
for col in toxicity_cols:
    toxic_texts = train_df[train_df[col] == 1]
    non_toxic_texts = train_df[train_df[col] == 0]
    if len(toxic_texts) > 0:
        print(f"  {col}:")
        print(f"    有毒评论: {toxic_texts['text_length'].mean():.2f} 字符")
        print(f"    正常评论: {non_toxic_texts['text_length'].mean():.2f} 字符")

# 可视化文本长度分布
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(toxicity_cols):
    toxic_lengths = train_df[train_df[col] == 1]['text_length']
    normal_lengths = train_df[train_df[col] == 0]['text_length']
    
    axes[idx].hist(normal_lengths, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    if len(toxic_lengths) > 0:
        axes[idx].hist(toxic_lengths, bins=50, alpha=0.7, label='Toxic', color='red', density=True)
    axes[idx].set_title(f'{col}', fontweight='bold')
    axes[idx].set_xlabel('Text Length (characters)')
    axes[idx].set_ylabel('Density')
    axes[idx].legend()
    axes[idx].set_xlim(0, 500)

plt.tight_layout()
plt.savefig('text_length_distribution.png', dpi=150, bbox_inches='tight')
print("\n📊 文本长度分布图已保存: text_length_distribution.png")

# 2.3 最常见词汇分析
print("\n2.3 各类别的最常见词汇:")

def get_top_words(texts, n=10):
    """获取最常见的词汇"""
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        all_words.extend(words)
    return Counter(all_words).most_common(n)

# 停用词
stop_words = {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'can', 'you', 'all', 'were', 'they', 'we', 'when', 'your', 'said', 'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very', 'what', 'know', 'just', 'first', 'get', 'over', 'think', 'also', 'its', 'after', 'back', 'other', 'many', 'than', 'only', 'those', 'come', 'day', 'most', 'us', 'go', 'see', 'now', 'way', 'who', 'did', 'my', 'no', 'work', 'may', 'well', 'should', 'any', 'same', 'such', 'take', 'were', 'me', 'even', 'here', 'use', 'an', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very', 'what', 'know', 'just', 'first', 'get', 'over', 'think', 'also', 'its', 'after', 'back', 'other', 'many', 'than', 'only', 'those', 'come', 'day', 'most', 'us', 'go', 'see', 'now', 'way', 'who', 'did', 'my', 'no', 'work', 'may', 'well', 'should', 'any', 'same', 'such', 'take', 'were', 'me', 'even', 'here', 'use', 'an'}

for col in toxicity_cols[:3]:  # 只显示前3个类别
    toxic_texts = train_df[train_df[col] == 1]['comment_text'].tolist()
    normal_texts = train_df[train_df[col] == 0]['comment_text'].tolist()
    
    toxic_words = [(w, c) for w, c in get_top_words(toxic_texts, 20) if w not in stop_words][:10]
    normal_words = [(w, c) for w, c in get_top_words(normal_texts, 20) if w not in stop_words][:10]
    
    print(f"\n{col.upper()}:")
    print(f"  有毒评论常见词: {[w for w, c in toxic_words]}")
    print(f"  正常评论常见词: {[w for w, c in normal_words]}")

print("\n" + "="*60)
print("✅ 数据分析部分完成!")
print("="*60)

plt.show() if 'plt' in dir() else None
