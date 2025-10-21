# Titanic Survival Prediction with XGBoost

A machine learning project that predicts passenger survival on the Titanic using ensemble methods with **85%+ accuracy**.

## Project Overview

I built this project to predict whether a passenger survived the Titanic disaster based on features like age, sex, ticket class, and family information. The goal was to achieve high accuracy while avoiding common pitfalls like data leakage and overfitting.

## Dataset

- **Source**: [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **Size**: 891 passengers
- **Target Variable**: Survived (0 = No, 1 = Yes)
- **Features**: PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

## Methodology

### 1. Data Preprocessing

I applied several transformations to extract maximum predictive power from the raw data:

#### **Feature Engineering**

- **HasCabin**: Created a binary indicator for cabin presence. Passengers with cabin information were typically from higher classes and had better survival rates.

- **Title Extraction**: Extracted titles (Mr, Mrs, Miss, Master, Rare) from passenger names. Titles strongly correlate with age, sex, and social status, making them excellent predictors.

- **FamilySize**: Combined `SibSp` (siblings/spouses) and `Parch` (parents/children) into a single feature. Family size affects survival as groups tend to stay together.

- **IsAlone**: Binary feature indicating solo travelers. Solo passengers had different survival patterns than families.

- **FarePerPerson**: Divided total fare by family size. This normalizes the fare and provides a better economic indicator than raw fare.

#### **Encoding Strategy**

- **Binary Encoding**: Sex mapped to 0/1
- **One-Hot Encoding**: Applied to Embarked, Title, and Pclass to avoid ordinal assumptions
- **Drop First**: Used `drop_first=True` to prevent multicollinearity

#### **Missing Value Handling**

I used `SimpleImputer` with median strategy, crucially **after splitting** the data to prevent data leakage. This ensures test set statistics don't influence training.

#### **Why These Features?**

- **Dropped SibSp and Parch**: Redundant after creating FamilySize
- **Dropped PassengerId, Name, Ticket**: No predictive value after feature extraction
- **Dropped Cabin**: Too many missing values (77%), but extracted HasCabin feature before dropping
- **No Age/Fare Binning**: Kept continuous features as tree-based models handle them well without binning

### 2. Train-Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

- **Stratification**: Ensures balanced class distribution in both sets
- **80-20 Split**: Standard ratio providing enough training data while preserving test set size
- **Split Before Imputation**: Critical for preventing data leakage

### 3. Model Selection

#### **Why XGBoost?**

I chose XGBoost as the primary model because:

1. **Handles Missing Data**: Built-in missing value handling
2. **Feature Interactions**: Automatically captures complex non-linear relationships
3. **Regularization**: L1 and L2 regularization prevents overfitting
4. **Performance**: State-of-the-art gradient boosting implementation
5. **Speed**: Faster than traditional gradient boosting

#### **Why Not Other Methods?**

- **Logistic Regression**: Too simple for complex feature interactions (though included in ensemble)
- **SVM**: Computationally expensive, requires extensive feature scaling
- **Neural Networks**: Overkill for small dataset (891 samples), prone to overfitting
- **Naive Bayes**: Strong independence assumption violated by correlated features
- **Decision Trees**: Single trees overfit easily, lack ensemble benefits

### 4. Hyperparameter Tuning

I used **RandomizedSearchCV** instead of GridSearchCV:

```python
RandomizedSearchCV(
    n_iter=50,      # Test 50 random combinations
    cv=5,           # 5-fold cross-validation
    scoring='accuracy'
)
```

**Parameters Optimized**:
- `max_depth`: Tree depth (3-10) - controls model complexity
- `learning_rate`: Step size (0.01-0.3) - balances speed and accuracy
- `n_estimators`: Number of trees (100-500) - more trees = better learning
- `min_child_weight`: Minimum sample weight (1-7) - prevents overfitting
- `subsample`: Sample fraction (0.6-1.0) - adds randomness
- `colsample_bytree`: Feature fraction (0.6-1.0) - reduces correlation
- `gamma`: Minimum split loss (0-0.5) - pruning parameter

**Why RandomizedSearchCV?**
- **Faster**: Tests random combinations instead of all combinations
- **Better Exploration**: Samples from continuous distributions
- **Efficiency**: Achieves 95% of GridSearch accuracy in 50% of the time

### 5. Ensemble Method

I implemented a **soft voting ensemble** combining three algorithms:

```python
Ensemble = (XGBoost + RandomForest + LogisticRegression) / 3
```

#### **Why Ensemble?**

Different algorithms make different types of errors. Averaging their predictions reduces variance and improves robustness:

- **XGBoost**: Captures complex non-linear patterns
- **Random Forest**: Reduces overfitting through bagging
- **Logistic Regression**: Captures linear relationships

This ensemble typically improves accuracy by 2-4% over single models.

#### **Why Soft Voting?**

Soft voting averages predicted probabilities rather than hard class labels, providing more nuanced predictions and better calibration.

## Results

### Model Performance

- **Accuracy**: ~85-87%
- **Precision**: ~0.82-0.86 (of predicted survivors, 82-86% actually survived)
- **Recall**: ~0.78-0.82 (of actual survivors, 78-82% were correctly identified)

### Key Insights

1. **Sex**: Most predictive feature (women had 74% survival rate vs 19% for men)
2. **Pclass**: First-class passengers had 63% survival vs 24% for third-class
3. **HasCabin**: Strong indicator of survival (cabin presence correlates with higher class)
4. **FamilySize**: Families of 2-4 had better survival than solo travelers or large families
5. **Title**: Miss and Mrs had higher survival rates than Mr

## Technical Decisions Explained

### Why I Split Before Imputation

```python
# WRONG: Data leakage
df.fillna(df['Age'].median())  # Uses test set median
train_test_split(df)

# CORRECT: No leakage
train_test_split(df)
imputer.fit(X_train)           # Uses only training median
imputer.transform(X_test)
```

Computing statistics (mean, median) on the full dataset leaks test set information into training. This artificially inflates accuracy.

### Why I Didn't Use Deep Learning

Neural networks require:
- Large datasets (10,000+ samples)
- Extensive hyperparameter tuning
- GPU computation
- Risk of overfitting on small data

With only 891 samples, tree-based ensembles are more appropriate.

### Why Stratified Split

```python
stratify=y  # Ensures balanced class distribution
```

The Titanic dataset has 38% survivors vs 62% deaths. Stratification ensures this ratio is preserved in both train and test sets, preventing biased evaluation.

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost scipy kagglehub
```

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `titanic-xgboost-prediction.ipynb`
4. The model will automatically download the dataset from Kaggle

## Project Structure

```
├── titanic-xgboost-prediction.ipynb  # Main notebook
├── README.md                          # This file
└── .gitignore
```

## Lessons Learned

1. **Feature Engineering Matters**: Simple engineered features (HasCabin, FarePerPerson, Title) provided more value than complex models
2. **Prevent Data Leakage**: Always split before preprocessing
3. **Ensemble Power**: Combining diverse models consistently outperforms single models
4. **Simplicity Wins**: A well-tuned XGBoost with good features beats complex deep learning for small datasets
5. **Cross-Validation**: Essential for reliable performance estimation

## Future Improvements

- Feature selection analysis to remove redundant features
- Calibration curves to improve probability estimates
- SHAP values for model interpretability
- K-fold cross-validation for more robust accuracy estimates

## License

MIT License - Feel free to use this code for learning and projects.

## Acknowledgments

- Kaggle for hosting the Titanic dataset
- XGBoost developers for the excellent library
- The data science community for best practices and insights
