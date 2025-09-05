# Bagging Implementation from Scratch

## Overview
This project demonstrates the Bagging (Bootstrap Aggregating) ensemble learning technique using Decision Trees on the Iris dataset. The implementation shows how multiple weak learners (Decision Trees) can be combined to create a more robust and accurate classifier through bootstrap sampling and majority voting.

## Features
- **Bootstrap Sampling**: Implements bagging by sampling with replacement from training data
- **Multiple Decision Trees**: Creates multiple Decision Tree classifiers on different bootstrap samples
- **Ensemble Prediction**: Demonstrates how individual tree predictions are combined
- **Visualization**: Includes decision tree visualization and decision boundary plots
- **Iris Dataset**: Uses the classic Iris dataset for binary classification (2 species)
- **From Scratch Implementation**: Manually implements bagging without using sklearn's BaggingClassifier

## Requirements
```python
numpy
pandas
matplotlib
seaborn
scikit-learn
mlxtend
```

## Installation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend
```

## Usage

### 1. Dataset Preparation
The code works with the Iris dataset:
- Loads Iris.csv from local directory
- Preprocesses data using LabelEncoder for species classification
- Filters to use only 2 species (binary classification)
- Selects 2 features: SepalWidthCm and PetalLengthCm
- Creates train/validation/test splits

### 2. Bootstrap Sampling
For each bagged tree:
- Samples 8 instances with replacement from 10 training samples
- Creates different bootstrap samples for each tree
- Each tree sees a different subset of the training data

### 3. Decision Tree Training
- Trains individual Decision Tree classifiers on each bootstrap sample
- Each tree learns different decision boundaries
- Trees may overfit to their specific bootstrap sample

### 4. Ensemble Prediction
- Makes predictions using individual trees
- Demonstrates how different trees can give different predictions
- Shows the concept of majority voting (though not fully implemented)

## Code Structure

### Main Components

#### Data Preparation
- **Dataset Loading**: Reads Iris.csv and preprocesses for binary classification
- **Feature Selection**: Uses SepalWidthCm and PetalLengthCm as features
- **Data Splitting**: Creates train (10 samples), validation (5 samples), and test (5 samples) sets

#### Bootstrap Sampling
```python
df_bag = df_train.sample(8, replace=True)
```
- Samples 8 instances with replacement from training data
- Each bootstrap sample is different, creating diversity in the ensemble

#### Decision Tree Training
```python
dt_bag = DecisionTreeClassifier()
dt_bag.fit(X, y)
```
- Trains individual Decision Tree on each bootstrap sample
- Each tree learns different patterns from its specific sample

#### Evaluation Function
```python
def evaluate(clf, X, y):
    clf.fit(X, y)
    plot_tree(clf)
    plot_decision_regions(X.values, y.values, clf=clf, legend=2)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
```
- Fits classifier and visualizes decision tree
- Plots decision regions
- Evaluates accuracy on validation set

## Mathematical Foundation

### Bootstrap Aggregating (Bagging)
Bagging reduces variance by:
1. **Bootstrap Sampling**: Create B bootstrap samples from training data
2. **Parallel Training**: Train B models on each bootstrap sample
3. **Aggregation**: Combine predictions using majority vote (classification) or average (regression)

### Decision Tree Diversity
Each tree in the ensemble is different because:
- Different bootstrap samples
- Different feature splits
- Different decision boundaries
- Reduces overfitting through ensemble diversity

## Visualization

The code generates several visualizations:
1. **Data Scatter Plot**: Shows the 2D feature space with species colors
2. **Decision Tree Structure**: Visualizes the tree structure for each bagged tree
3. **Decision Regions**: Shows how each tree divides the feature space
4. **Prediction Comparison**: Demonstrates different predictions from different trees

## Key Results

### Individual Tree Performance
- **Tree 1**: 40% accuracy on validation set
- **Tree 2**: 100% accuracy on validation set  
- **Tree 3**: 100% accuracy on validation set

### Prediction Diversity
For test point [2.2, 5.0]:
- **Tree 1**: Predicts class 1
- **Tree 2**: Predicts class 2
- **Tree 3**: Predicts class 2

This demonstrates how different trees can make different predictions, which is the foundation of ensemble learning.

## Bagging Benefits

1. **Reduced Variance**: Multiple models reduce overfitting
2. **Improved Generalization**: Better performance on unseen data
3. **Robustness**: Less sensitive to outliers and noise
4. **Parallel Training**: Trees can be trained independently
5. **Bias-Variance Trade-off**: Reduces variance without increasing bias

## Limitations

- **Small Dataset**: Limited to 10 training samples for demonstration
- **Binary Classification**: Only 2 species from Iris dataset
- **No Majority Voting**: Doesn't implement final ensemble prediction
- **Limited Trees**: Only 3 trees in the ensemble
- **No Cross-Validation**: Simple train/validation split

## Applications

This implementation is useful for:
- Understanding ensemble learning concepts
- Learning bootstrap sampling techniques
- Visualizing decision tree diversity
- Educational purposes in machine learning
- Demonstrating bagging fundamentals

## Future Improvements

- Implement majority voting for final predictions
- Add more trees to the ensemble
- Use larger training dataset
- Implement out-of-bag (OOB) error estimation
- Add feature importance analysis
- Implement parallel training
- Add cross-validation
- Support for multi-class classification
- Implement sklearn's BaggingClassifier for comparison

## Educational Value

This notebook demonstrates:
- How bootstrap sampling creates diversity
- Why ensemble methods work
- Decision tree visualization
- The concept of weak learners
- Variance reduction through aggregation
- Practical implementation of bagging

## Author
Created as part of machine learning projects to understand ensemble learning concepts.

## License
This project is for educational purposes.
