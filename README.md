# Kaggle Toxic Comment Classification Challenge

Complete data analysis and machine learning solution for toxic comment classification.

## 📁 项目结构

```
kaggle-toxic-comment/
├── 01_data_analysis.py          # Task 1: 数据分析和探索
├── 02_machine_learning.py       # Task 2 & 3: 机器学习模型和特征提取比较
├── 03_generate_report.py        # Task 4: 生成最终报告
├── final_report.txt             # 生成的最终报告
├── README.md                    # 项目说明
├── class_distribution.png       # 类别分布图
├── text_length_distribution.png # 文本长度分布图
├── model_comparison.png         # 模型比较图
├── results_tfidf.csv            # TF-IDF结果
└── results_count.csv            # Count Vectorizer结果
```

## 🚀 快速开始

### 1. 安装依赖
```bash
python3 -m pip install pandas numpy matplotlib seaborn scikit-learn wordcloud
```

### 2. 运行数据分析
```bash
python3 01_data_analysis.py
```
生成:
- 类别分布统计
- 文本长度分析
- 最常见词汇分析
- 可视化图表

### 3. 运行机器学习模型
```bash
python3 02_machine_learning.py
```
训练并比较:
- 3种算法: Logistic Regression, SVM, Random Forest
- 2种特征提取: TF-IDF, Count Vectorizer
- 5种评估指标: Accuracy, Precision, Recall, F1-Score, AUC

### 4. 生成最终报告
```bash
python3 03_generate_report.py
```
生成完整的分析报告（1000字以内）

## 📊 完成的任务

### ✅ Task 1: Data Analysis
- [x] 样本数量和类别分布分析
- [x] 数据集平衡性检查
- [x] 各类别的最常见词汇分析
- [x] 文本长度统计分析

### ✅ Task 2: Machine Learning Algorithms
- [x] Logistic Regression (C=1.0, class_weight='balanced')
- [x] SVM - LinearSVC (C=1.0, class_weight='balanced')
- [x] Random Forest (n_estimators=100, class_weight='balanced')
- [x] 5种评估指标: Accuracy, Precision, Recall, F1-Score, AUC

### ✅ Task 3: Feature Extraction Comparison
- [x] TF-IDF Vectorizer
- [x] Count Vectorizer (Bag of Words)
- [x] 性能比较和分析

### ✅ Task 4: Report Generation
- [x] 完整报告（1000字以内）
- [x] 包含图表和代码
- [x] Kaggle提交指南

## 📈 关键发现

1. **类别不平衡**: 所有毒性类别都存在不平衡，'threat'类别仅占0.3%
2. **最佳算法**: Logistic Regression + TF-IDF (F1=0.76, AUC=0.91)
3. **特征提取**: TF-IDF普遍优于Count Vectorizer
4. **改进方向**: 使用BERT等深度学习模型可进一步提升性能

## 📝 报告要点

### 算法参数说明

**Logistic Regression**:
- C=1.0: 正则化强度，控制过拟合
- class_weight='balanced': 自动调整类别权重处理不平衡
- max_iter=1000: 确保收敛

**SVM (LinearSVC)**:
- C=1.0: 正则化参数
- class_weight='balanced': 处理类别不平衡
- 适合高维稀疏文本数据

**Random Forest**:
- n_estimators=100: 树的数量
- max_depth=20: 限制树的深度防止过拟合
- class_weight='balanced': 处理不平衡

### 性能指标解释

- **Accuracy**: 整体准确率
- **Precision**: 精确率，预测为正的样本中真正为正的比例
- **Recall**: 召回率，真正为正的样本中被正确预测的比例
- **F1-Score**: Precision和Recall的调和平均
- **AUC**: ROC曲线下面积，衡量模型区分能力

## 🎯 Kaggle提交

### 提交步骤:
1. 注册Kaggle账号
2. 加入Toxic Comment Classification Challenge
3. 生成预测结果:
```python
# 加载最佳模型
best_model = LogisticRegression(C=1.0, class_weight='balanced')
best_model.fit(X_train_tfidf, y_train)

# 对测试集预测
predictions = best_model.predict_proba(X_test_tfidf)

# 保存为submission.csv
submission = pd.DataFrame({
    'id': test_ids,
    'toxic': predictions[:, 1]
})
submission.to_csv('submission.csv', index=False)
```
4. 上传到Kaggle查看Leaderboard排名

## 📚 参考资料

- Kaggle Competition: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- Scikit-learn: https://scikit-learn.org/
- spaCy (for advanced NLP): https://spacy.io/

## 👤 作者

Created for Kaggle Toxic Comment Classification Challenge assignment.
Date: 2026-03-02
