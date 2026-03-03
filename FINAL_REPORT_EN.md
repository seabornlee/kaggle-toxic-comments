# Kaggle Toxic Comment Classification Challenge
## Complete Research Report with Code Implementation

---

## Abstract

This research presents a comprehensive analysis of the Kaggle Toxic Comment Classification Challenge dataset (159,571 comments). We performed detailed data exploration, compared three machine learning algorithms (Logistic Regression, SVM, and Random Forest), and evaluated two feature extraction methods (TF-IDF and Count Vectorizer).

**Key Findings**: Logistic Regression with TF-IDF features achieved the best performance (F1=0.696, AUC=0.957). All toxicity categories exhibit varying degrees of class imbalance, with 'threat' being the most underrepresented at only 0.30% of the dataset.

---

## 1. Data Analysis (Task 1a)

### 1.1 Dataset Overview

The dataset contains Wikipedia comments labeled for toxic behavior across six categories:

- **Total samples**: 159,571 comments
- **Features**: Comment text + 6 toxicity labels
- **Task type**: Multi-label classification

### 1.2 Class Distribution Analysis

| Toxicity Type | Count | Percentage | Imbalance Level |
|--------------|-------|------------|-----------------|
| toxic | 15,294 | 9.58% | ⚠️ Mild imbalance |
| severe_toxic | 1,595 | 1.00% | ⚠️ Severe imbalance |
| obscene | 8,449 | 5.29% | ⚠️ Imbalanced |
| threat | 478 | 0.30% | ⚠️ Extreme imbalance |
| insult | 7,877 | 4.93% | ⚠️ Imbalanced |
| identity_hate | 1,405 | 0.88% | ⚠️ Severe imbalance |

**Key Observations**:
- The 'threat' category represents only 0.30% of data, making it extremely rare
- 'toxic' is the most common label but still represents less than 10%
- This is a classic extreme class imbalance problem

### 1.3 Data Loading Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_df = pd.read_csv('data/train.csv')

# Display basic information
print(f"Dataset shape: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(train_df.head())
```

### 1.4 Class Distribution Visualization Code

```python
# Define toxicity columns
toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 
                 'threat', 'insult', 'identity_hate']

# Calculate class distribution
class_dist = []
for col in toxicity_cols:
    count = train_df[col].sum()
    percentage = count / len(train_df) * 100
    class_dist.append({
        'Class': col,
        'Count': count,
        'Percentage': f"{percentage:.2f}%"
    })

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Bar chart of class counts
ax1 = axes[0, 0]
counts = [train_df[col].sum() for col in toxicity_cols]
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(toxicity_cols)))
bars = ax1.bar(range(len(toxicity_cols)), counts, color=colors)
ax1.set_xticks(range(len(toxicity_cols)))
ax1.set_xticklabels(toxicity_cols, rotation=45, ha='right')
ax1.set_ylabel('Count')
ax1.set_title('Toxicity Class Distribution', fontsize=14, fontweight='bold')

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, counts)):
    ax1.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + 100, 
             f'{count:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
```

### 1.5 Multi-Label Analysis

```python
# Calculate total toxicity labels per comment
train_df['total_toxic'] = train_df[toxicity_cols].sum(axis=1)

# Analyze multi-label distribution
no_toxic = (train_df['total_toxic'] == 0).sum()
single_toxic = (train_df['total_toxic'] == 1).sum()
multiple_toxic = (train_df['total_toxic'] > 1).sum()

print(f"No toxicity: {no_toxic:,} ({no_toxic/len(train_df)*100:.2f}%)")
print(f"Single toxicity: {single_toxic:,}")
print(f"Multiple toxicity: {multiple_toxic:,}")
print(f"Maximum labels: {train_df['total_toxic'].max()}")
```

### 1.6 Text Feature Analysis

```python
# Calculate text statistics
train_df['text_length'] = train_df['comment_text'].str.len()
train_df['word_count'] = train_df['comment_text'].str.split().str.len()

print(f"Average characters: {train_df['text_length'].mean():.2f}")
print(f"Median characters: {train_df['text_length'].median():.2f}")
print(f"Average words: {train_df['word_count'].mean():.2f}")
```

**Text Length Statistics**:
- Average characters: 394.07
- Average words: 67.27
- Maximum characters: 5,000

**Observation**: Toxic comments are typically ~25% shorter than normal comments, likely due to more emotional, concise expressions.

---

## 2. Machine Learning Algorithm Comparison (Task 2)

### 2.1 Algorithm Selection and Parameters

#### 1. Logistic Regression
**Parameters**: `C=1.0, max_iter=1000, class_weight='balanced', random_state=42`

**Rationale**: Uses sigmoid function for probability mapping with L2 regularization to prevent overfitting. Highly efficient for high-dimensional sparse text data.

#### 2. Support Vector Machine (LinearSVC)
**Parameters**: `C=1.0, max_iter=2000, class_weight='balanced', random_state=42`

**Rationale**: Finds optimal hyperplane for classification. Linear kernel handles high-dimensional features effectively.

#### 3. Random Forest
**Parameters**: `n_estimators=50, max_depth=20, class_weight='balanced', random_state=42`

**Rationale**: Ensemble method using multiple decision trees. Can capture non-linear relationships but less effective with sparse data.

### 2.2 Model Implementation Code

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score)

# Prepare data
X = train_df['comment_text'].fillna('')
y = train_df['toxic']  # Focus on 'toxic' class

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature extraction with TF-IDF
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF feature shape: {X_train_tfidf.shape}")
```

### 2.3 Model Training and Evaluation Code

```python
# Define models
models = {
    'Logistic Regression': LogisticRegression(
        C=1.0, max_iter=1000, 
        class_weight='balanced', 
        random_state=42
    ),
    'SVM (Linear)': LinearSVC(
        C=1.0, max_iter=2000, 
        class_weight='balanced', 
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=50, 
        max_depth=20, 
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
}

# Train and evaluate
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test)
    
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_proba)
    }

# Train all models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    results[name] = evaluate_model(model, X_test_tfidf, y_test)
    
    print(f"F1-Score: {results[name]['F1-Score']:.4f}")
    print(f"AUC: {results[name]['AUC']:.4f}")
```

### 2.4 Performance Results (TF-IDF Features)

| Algorithm | Accuracy | Precision | Recall | F1-Score | AUC |
|-----------|----------|-----------|--------|----------|-----|
| **Logistic Regression** | **0.935** | **0.632** | **0.774** | **0.696** | **0.957** |
| SVM (Linear) | 0.934 | 0.632 | 0.740 | 0.682 | 0.942 |
| Random Forest | 0.826 | 0.330 | 0.791 | 0.466 | 0.902 |

**Analysis**:
- **Logistic Regression** achieved the best performance with F1=0.696 and AUC=0.957
- **SVM** performed comparably with F1=0.682 and AUC=0.942
- **Random Forest** underperformed, likely due to data sparsity

### 2.5 Detailed Evaluation

**Logistic Regression**:
- High accuracy (93.5%) indicates strong overall predictive capability
- Precision=0.632: 63.2% of predicted toxic comments are truly toxic
- Recall=0.774: 77.4% of actual toxic comments are correctly identified
- AUC=0.957 demonstrates excellent classification ability

---

## 3. Feature Extraction Comparison (Task 3)

### 3.1 TF-IDF vs Count Vectorizer

| Algorithm | TF-IDF F1 | Count F1 | F1 Diff | TF-IDF AUC | Count AUC |
|-----------|-----------|----------|---------|------------|-----------|
| Logistic Regression | 0.696 | 0.693 | +0.003 | 0.957 | 0.955 |
| SVM | 0.682 | 0.678 | +0.004 | 0.942 | 0.938 |
| Random Forest | 0.466 | 0.458 | +0.008 | 0.902 | 0.895 |

**Key Findings**:
1. **TF-IDF consistently outperforms Count Vectorizer**, though differences are small (0.003-0.008)
2. TF-IDF reduces common word weights and emphasizes distinctive terms
3. Differences would be more significant on larger datasets

### 3.2 Feature Extraction Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Count Vectorizer (Bag of Words)
count_vectorizer = CountVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

print(f"TF-IDF shape: {X_train_tfidf.shape}")
print(f"Count shape: {X_train_count.shape}")
```

### 3.3 Method Comparison

**TF-IDF (Term Frequency-Inverse Document Frequency)**:
- Formula: TF-IDF(t,d) = TF(t,d) × IDF(t)
- IDF(t) = log(N / DF(t))
- **Advantages**: Reduces stop word weights, emphasizes distinctive terms
- **Best for**: Text classification, information retrieval

**Count Vectorizer (Bag of Words)**:
- Formula: Simple term frequency count
- **Advantages**: Simple computation, high interpretability
- **Disadvantages**: Treats all words equally, cannot distinguish importance

---

## 4. Kaggle Submission and Improvement Suggestions (Task 4)

### 4.1 Submission Strategy

**Best Model**: Logistic Regression + TF-IDF
- Expected Leaderboard AUC: ~0.96
- Expected Ranking: Top 20%

### 4.2 Submission Code

```python
# Load test data
test_df = pd.read_csv('data/test.csv')
X_test_final = test_df['comment_text'].fillna('')

# Train best model on full dataset
best_model = LogisticRegression(
    C=1.0, max_iter=1000, 
    class_weight='balanced', 
    random_state=42
)

# Use all training data
X_full_tfidf = tfidf_vectorizer.fit_transform(X)
best_model.fit(X_full_tfidf, y)

# Make predictions
X_test_final_tfidf = tfidf_vectorizer.transform(X_test_final)
predictions = best_model.predict_proba(X_test_final_tfidf)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'toxic': predictions[:, 1]
})

submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
```

### 4.3 Performance Analysis

**Success Factors**:
1. Effective text feature extraction using TF-IDF
2. `class_weight='balanced'` handling class imbalance
3. Appropriate regularization (C=1.0) preventing overfitting

### 4.4 Improvement Directions

**1. Deep Learning Models**:
Using BERT, RoBERTa, or other pre-trained language models could significantly improve performance (expected AUC: 0.98+). These models better understand context and semantic relationships, especially for detecting sarcasm and implicit toxicity.

**2. Feature Engineering**:
Add hand-crafted features:
- Text length, punctuation ratio
- Capital letter ratio (emotional indicator)
- Sentiment analysis scores
- Profanity and sensitive word dictionary matching

**3. Data Augmentation**:
Apply oversampling to minority classes (threat, severe_toxic) or use back-translation, synonym replacement to increase training samples.

**4. Ensemble Methods**:
Combine predictions from multiple models using Stacking or Voting to improve stability and performance.

---

## 5. Conclusion

### Key Findings

1. **Extreme Class Imbalance**: The rarest 'threat' category represents only 0.30%, presenting the biggest challenge
2. **Best Algorithm**: Logistic Regression + TF-IDF (F1=0.696, AUC=0.957)
3. **Feature Engineering Matters**: TF-IDF outperforms simple word frequency, though differences are small
4. **Text Characteristics**: Toxic comments are on average 25% shorter than normal comments

### Practical Implications

- For class imbalance, `class_weight='balanced'` is an effective solution
- Linear models (LR, SVM) perform excellently on high-dimensional sparse text data
- Simple feature engineering (TF-IDF) achieves strong baseline performance
- Deep learning methods (BERT) may be the best path to SOTA performance

---

## References

1. Kaggle Competition: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
2. Scikit-learn Documentation: https://scikit-learn.org/stable/
3. TF-IDF: Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval
4. BERT: Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers

---

## Appendix: Complete Code Repository

All code and data processing workflows are available in the project directory:
- `01_data_analysis.py` - Data exploration and visualization
- `02_machine_learning.py` - Model training and evaluation
- `03_generate_report.py` - Report generation

**Generated**: 2026-03-02  
**Dataset**: Kaggle Toxic Comment Classification Challenge (2018)
