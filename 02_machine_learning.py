#!/usr/bin/env python3
"""
Kaggle Toxic Comment Classification Challenge
机器学习模型训练和评估
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report,
                           confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("机器学习模型训练和评估")
print("="*60)

# ==================== 1. 数据准备 ====================
print("\n📥 1. 数据准备...")

# 加载数据（实际使用时替换为真实数据）
# train_df = pd.read_csv('data/train.csv')

# 使用模拟数据
np.random.seed(42)
n_samples = 3000  # 减少样本数以加快训练

train_df = pd.DataFrame({
    'comment_text': [
        'This is a great article thank you for sharing' if np.random.random() > 0.2
        else 'You are stupid and dumb' if np.random.random() > 0.5
        else 'I hate this so much terrible' if np.random.random() > 0.5
        else 'Nice work keep it up'
        for _ in range(n_samples)
    ],
    'toxic': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
})

# 为了简化，我们只预测'toxic'类别
toxicity_col = 'toxic'

X = train_df['comment_text']
y = train_df[toxicity_col]

print(f"✅ 数据准备完成: {len(X)} 条评论")
print(f"正样本比例: {y.mean():.2%}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")

# ==================== 2. 特征提取方法 ====================
print("\n" + "="*60)
print("📊 2. 特征提取方法比较")
print("="*60)

# 2.1 TF-IDF
print("\n🔤 2.1 TF-IDF 特征提取...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),  # 使用unigram和bigram
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"✅ TF-IDF特征维度: {X_train_tfidf.shape}")

# 2.2 Count Vectorizer (Bag of Words)
print("\n📝 2.2 Count Vectorizer (词袋模型)...")
count_vectorizer = CountVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

print(f"✅ Count特征维度: {X_train_count.shape}")

# ==================== 3. 模型定义 (Task 2) ====================
print("\n" + "="*60)
print("🤖 3. 机器学习模型定义")
print("="*60)

# 选择3种算法: Logistic Regression, SVM, Random Forest
models = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'),
        'params': 'C=1.0, max_iter=1000, class_weight=balanced'
    },
    'SVM (Linear)': {
        'model': LinearSVC(C=1.0, max_iter=2000, class_weight='balanced'),
        'params': 'C=1.0, max_iter=2000, class_weight=balanced'
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=100, max_depth=20, 
                                       random_state=42, class_weight='balanced'),
        'params': 'n_estimators=100, max_depth=20, class_weight=balanced'
    }
}

print("\n选择的3种算法及其参数:")
for name, info in models.items():
    print(f"\n{name}:")
    print(f"  参数: {info['params']}")

# ==================== 4. 模型训练和评估 ====================
print("\n" + "="*60)
print("🎯 4. 模型训练和评估 (使用 TF-IDF)")
print("="*60)

def evaluate_model(model, X_test, y_test, model_name):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    
    # 对于SVM，decision_function代替predict_proba
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba

results_tfidf = {}

for name, info in models.items():
    print(f"\n🔄 训练 {name}...")
    model = info['model']
    
    # 训练
    model.fit(X_train_tfidf, y_train)
    
    # 评估
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test_tfidf, y_test, name)
    results_tfidf[name] = metrics
    
    print(f"✅ {name} 训练完成")
    print(f"   Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   F1-Score: {metrics['F1-Score']:.4f}")
    print(f"   AUC: {metrics['AUC']:.4f}")

# ==================== 5. 特征提取方法比较 (Task 3) ====================
print("\n" + "="*60)
print("🔄 5. 特征提取方法比较 (TF-IDF vs Count Vectorizer)")
print("="*60)

results_count = {}

print("\n使用 Count Vectorizer (词袋模型) 重新训练...")

for name, info in models.items():
    print(f"\n🔄 训练 {name} (Count)...")
    model = info['model'].__class__(**info['model'].get_params())
    model.fit(X_train_count, y_train)
    
    metrics, _, _ = evaluate_model(model, X_test_count, y_test, name)
    results_count[name] = metrics
    
    print(f"✅ F1-Score: {metrics['F1-Score']:.4f}")

# ==================== 6. 结果比较和可视化 ====================
print("\n" + "="*60)
print("📊 6. 结果比较")
print("="*60)

# 创建比较表格
comparison_data = []
for model_name in models.keys():
    comparison_data.append({
        'Model': model_name,
        'TF-IDF_F1': results_tfidf[model_name]['F1-Score'],
        'Count_F1': results_count[model_name]['F1-Score'],
        'TF-IDF_AUC': results_tfidf[model_name]['AUC'],
        'Count_AUC': results_count[model_name]['AUC']
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nF1-Score 比较:")
print(comparison_df[['Model', 'TF-IDF_F1', 'Count_F1']].to_string(index=False))

print("\nAUC 比较:")
print(comparison_df[['Model', 'TF-IDF_AUC', 'Count_AUC']].to_string(index=False))

# 可视化比较
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# F1-Score 比较
ax1 = axes[0, 0]
x = np.arange(len(models))
width = 0.35
ax1.bar(x - width/2, [results_tfidf[m]['F1-Score'] for m in models.keys()], 
        width, label='TF-IDF', color='skyblue')
ax1.bar(x + width/2, [results_count[m]['F1-Score'] for m in models.keys()], 
        width, label='Count', color='lightcoral')
ax1.set_ylabel('F1-Score')
ax1.set_title('F1-Score Comparison by Feature Extraction Method', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models.keys(), rotation=15)
ax1.legend()
ax1.set_ylim(0, 1)

# AUC 比较
ax2 = axes[0, 1]
ax2.bar(x - width/2, [results_tfidf[m]['AUC'] for m in models.keys()], 
        width, label='TF-IDF', color='skyblue')
ax2.bar(x + width/2, [results_count[m]['AUC'] for m in models.keys()], 
        width, label='Count', color='lightcoral')
ax2.set_ylabel('AUC')
ax2.set_title('AUC Comparison by Feature Extraction Method', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models.keys(), rotation=15)
ax2.legend()
ax2.set_ylim(0, 1)

# 所有指标的雷达图（以最佳模型为例）
ax3 = axes[1, 0]
best_model = max(results_tfidf.keys(), key=lambda x: results_tfidf[x]['F1-Score'])
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
values = [results_tfidf[best_model][m] for m in metrics]
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
values += values[:1]
angles += angles[:1]

ax3.plot(angles, values, 'o-', linewidth=2, label=f'{best_model} (TF-IDF)')
ax3.fill(angles, values, alpha=0.25)
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(metrics)
ax3.set_ylim(0, 1)
ax3.set_title(f'Performance Metrics - {best_model}', fontweight='bold')
ax3.legend()
ax3.grid(True)

# 混淆矩阵（以最佳模型为例）
ax4 = axes[1, 1]
best_model_obj = models[best_model]['model']
best_model_obj.fit(X_train_tfidf, y_train)
y_pred = best_model_obj.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_title(f'Confusion Matrix - {best_model}', fontweight='bold')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("\n📊 模型比较图已保存: model_comparison.png")

# ==================== 7. 详细报告 ====================
print("\n" + "="*60)
print("📋 7. 详细评估报告")
print("="*60)

print("\n" + "="*60)
print("TF-IDF 特征提取结果:")
print("="*60)
for name in models.keys():
    print(f"\n{name}:")
    for metric, value in results_tfidf[name].items():
        print(f"  {metric:12}: {value:.4f}")

print("\n" + "="*60)
print("Count Vectorizer (词袋模型) 结果:")
print("="*60)
for name in models.keys():
    print(f"\n{name}:")
    for metric, value in results_count[name].items():
        print(f"  {metric:12}: {value:.4f}")

print("\n" + "="*60)
print("✅ 模型训练和评估完成!")
print("="*60)

# 保存结果到CSV
results_tfidf_df = pd.DataFrame(results_tfidf).T
results_tfidf_df.to_csv('results_tfidf.csv')
results_count_df = pd.DataFrame(results_count).T
results_count_df.to_csv('results_count.csv')
print("\n📁 结果已保存到: results_tfidf.csv, results_count.csv")

plt.show() if 'plt' in dir() else None
