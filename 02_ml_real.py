#!/usr/bin/env python3
"""
使用真实数据的机器学习模型训练和评估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🤖 真实数据 - 机器学习模型训练和评估")
print("="*70)

# 加载真实数据
print("\n📥 加载真实数据...")
train_df = pd.read_csv('data/train.csv')

# 为了加快训练速度，使用子集
print("使用20%数据子集进行训练...")
train_df = train_df.sample(frac=0.2, random_state=42)

print(f"✅ 数据加载完成: {len(train_df):,} 条评论")

# 本次专注于 'toxic' 类别的分类
X = train_df['comment_text'].fillna('')
y = train_df['toxic']

print(f"正样本比例: {y.mean():.2%}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {len(X_train):,} | 测试集: {len(X_test):,}")

# ==================== 特征提取 ====================
print("\n" + "="*70)
print("🔤 特征提取")
print("="*70)

# TF-IDF
print("\n提取 TF-IDF 特征...")
tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"TF-IDF特征维度: {X_train_tfidf.shape}")

# Count Vectorizer
print("\n提取 Count Vectorizer 特征...")
count_vec = CountVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2), min_df=2)
X_train_count = count_vec.fit_transform(X_train)
X_test_count = count_vec.transform(X_test)
print(f"Count特征维度: {X_train_count.shape}")

# ==================== 模型训练和评估 ====================
print("\n" + "="*70)
print("🤖 模型训练和评估")
print("="*70)

def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    """训练模型并评估"""
    print(f"\n🔄 训练 {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # 获取预测概率（用于AUC）
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"✅ {name} 完成!")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    return metrics, y_pred

# 定义模型
models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42),
    'SVM (Linear)': LinearSVC(C=1.0, max_iter=2000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
}

# 使用TF-IDF训练
print("\n" + "-"*70)
print("使用 TF-IDF 特征:")
print("-"*70)
results_tfidf = {}
for name, model in models.items():
    metrics, _ = train_and_evaluate(model, X_train_tfidf, X_test_tfidf, y_train, y_test, name)
    results_tfidf[name] = metrics

# 使用Count训练
print("\n" + "-"*70)
print("使用 Count Vectorizer 特征:")
print("-"*70)
results_count = {}
for name, model in models.items():
    # 重新初始化模型
    if name == 'Logistic Regression':
        model = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
    elif name == 'SVM (Linear)':
        model = LinearSVC(C=1.0, max_iter=2000, class_weight='balanced', random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=50, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
    
    metrics, _ = train_and_evaluate(model, X_train_count, X_test_count, y_train, y_test, name)
    results_count[name] = metrics

# ==================== 结果比较和可视化 ====================
print("\n" + "="*70)
print("📊 结果比较")
print("="*70)

# 创建比较表格
comparison_df = pd.DataFrame({
    'Model': list(models.keys()),
    'TF-IDF_F1': [results_tfidf[m]['F1-Score'] for m in models.keys()],
    'Count_F1': [results_count[m]['F1-Score'] for m in models.keys()],
    'TF-IDF_AUC': [results_tfidf[m]['AUC'] for m in models.keys()],
    'Count_AUC': [results_count[m]['AUC'] for m in models.keys()]
})

print("\nF1-Score 比较:")
print(comparison_df[['Model', 'TF-IDF_F1', 'Count_F1']].to_string(index=False))

print("\nAUC 比较:")
print(comparison_df[['Model', 'TF-IDF_AUC', 'Count_AUC']].to_string(index=False))

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# F1比较
ax1 = axes[0, 0]
x = np.arange(len(models))
width = 0.35
ax1.bar(x - width/2, [results_tfidf[m]['F1-Score'] for m in models.keys()], width, label='TF-IDF', color='skyblue')
ax1.bar(x + width/2, [results_count[m]['F1-Score'] for m in models.keys()], width, label='Count', color='lightcoral')
ax1.set_ylabel('F1-Score')
ax1.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models.keys(), rotation=15)
ax1.legend()
ax1.set_ylim(0, 1)

# AUC比较
ax2 = axes[0, 1]
ax2.bar(x - width/2, [results_tfidf[m]['AUC'] for m in models.keys()], width, label='TF-IDF', color='skyblue')
ax2.bar(x + width/2, [results_count[m]['AUC'] for m in models.keys()], width, label='Count', color='lightcoral')
ax2.set_ylabel('AUC')
ax2.set_title('AUC Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models.keys(), rotation=15)
ax2.legend()
ax2.set_ylim(0, 1)

# 所有指标对比（最佳模型）
ax3 = axes[1, 0]
best_model = max(results_tfidf.keys(), key=lambda x: results_tfidf[x]['F1-Score'])
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
values = [results_tfidf[best_model]['Accuracy'], 
          results_tfidf[best_model]['Precision'],
          results_tfidf[best_model]['Recall'],
          results_tfidf[best_model]['F1-Score'],
          results_tfidf[best_model]['AUC']]
ax3.bar(metrics_names, values, color='steelblue')
ax3.set_ylabel('Score')
ax3.set_title(f'{best_model} Performance (TF-IDF)', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 1)
for i, v in enumerate(values):
    ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

# 混淆矩阵
ax4 = axes[1, 1]
best_clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
best_clf.fit(X_train_tfidf, y_train)
y_pred = best_clf.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=False)
ax4.set_title(f'Confusion Matrix - {best_model}', fontsize=14, fontweight='bold')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('real_model_comparison.png', dpi=150, bbox_inches='tight')
print("\n📊 模型比较图已保存: real_model_comparison.png")

# 保存结果
results_tfidf_df = pd.DataFrame(results_tfidf).T
results_tfidf_df.to_csv('real_results_tfidf.csv')
results_count_df = pd.DataFrame(results_count).T
results_count_df.to_csv('real_results_count.csv')

print("\n📁 结果已保存:")
print("  - real_results_tfidf.csv")
print("  - real_results_count.csv")

print("\n" + "="*70)
print("✅ 机器学习模型训练和评估完成!")
print("="*70)

print(f"\n🏆 最佳模型: {best_model}")
print(f"   F1-Score: {results_tfidf[best_model]['F1-Score']:.4f}")
print(f"   AUC: {results_tfidf[best_model]['AUC']:.4f}")
