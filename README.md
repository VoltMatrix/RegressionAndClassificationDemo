# RegressionAndClassificationDemo

# Linear and Logistic Regression with Scikit-Learn

This repository contains Python code demonstrating linear regression, polynomial regression, regularized models, and logistic/softmax regression using Scikit-learn and NumPy. The code includes synthetic data for regression tasks and the Iris dataset for classification tasks, with visualizations for learning curves and classification probabilities.

## Overview
The code covers:
- **Linear Regression**: Using Normal Equation, SVD, Batch Gradient Descent, Stochastic Gradient Descent, and Mini-batch Gradient Descent (via SGDRegressor) on synthetic data (`y = 4 + 3x + noise`).
- **Polynomial Regression**: Fitting a quadratic relationship (`y = 0.5x^2 + x + 2 + noise`) using polynomial features.
- **Learning Curves**: Plotting training and validation errors to diagnose model performance.
- **Regularized Models**: Applying Ridge, Lasso, and Elastic Net regression to prevent overfitting.
- **Early Stopping**: Implementing early stopping with SGDRegressor on high-degree polynomial features.
- **Logistic Regression**: Binary classification on Iris dataset (Virginica vs. not Virginica) with probability visualization.
- **Softmax Regression**: Multi-class classification on Iris dataset using petal length and width.

## Requirements
- Python 3.x
- Scikit-learn
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install scikit-learn numpy matplotlib



