import numpy as np  # Import NumPy for numerical operations like arrays, matrix math, and random number generation
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet,SGDRegressor  # Import Scikit-Learn models for regression and classification
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # Import tools for polynomial feature creation and feature scaling
from sklearn.metrics import mean_squared_error  # Import function to compute Mean Squared Error for model evaluation
from sklearn.model_selection import train_test_split  # Import function to split data into training and validation sets
from sklearn.pipeline import Pipeline  # Import Pipeline to chain preprocessing and modeling steps
from sklearn import datasets  # Import datasets module to load the Iris dataset
import matplotlib.pyplot as plt  # Import Matplotlib for plotting learning curves and classification probabilities
from sklearn.base import clone  # Import clone to create copies of models without retraining

# Section 1: Linear Regression with Synthetic Data
# Purpose: Model a linear relationship y = 4 + 3x + noise using different optimization methods
# Why: To compare analytical (Normal Equation, SVD) and iterative (Gradient Descent) approaches
# What: Generate 100 data points with x in [0, 2], y = 4 + 3x + noise, and fit a linear model
np.random.seed(42)  # Set random seed for reproducibility of random numbers
X = 2 * np.random.rand(100, 1)  # Generate 100 random x values between 0 and 2, shape (100, 1) for a single feature
y = 4 + 3 * X + np.random.rand(100, 1)  # Compute y = 4 + 3x + noise (noise between 0 and 1), shape (100, 1)

# Normal Equation: theta = (X^T X)^(-1) X^T y
# Why: Provides an analytical solution to find optimal parameters (intercept and slope) by minimizing MSE
X_b = np.c_[np.ones((100, 1)), X]  # Add a column of 1s to X for the intercept, shape becomes (100, 2): [1, x]
X_b_T = X_b.T  # Compute transpose of X_b, shape (2, 100), for matrix multiplication in Normal Equation
X_b_T_dot_X_b = X_b_T.dot(X_b)  # Compute X^T X, shape (2, 2), part of the Normal Equation
inv_X_b_T_dot_X_b = np.linalg.inv(X_b_T_dot_X_b)  # Compute inverse of X^T X, shape (2, 2), for solving the equation
theta_best = inv_X_b_T_dot_X_b.dot(X_b_T).dot(y)  # Compute theta = (X^T X)^(-1) X^T y, shape (2, 1)
print("Normal Equation theta (intercept, slope):", theta_best.ravel())  # Flatten theta to 1D for printing, expect ~[4, 3]

# Predict for new inputs x = 0 and x = 2
X_new = np.array([[0], [2]])  # Create test inputs x = 0 and x = 2, shape (2, 1)
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add intercept column, shape (2, 2): [1, 0], [1, 2]
y_predict = X_new_b.dot(theta_best)  # Compute predictions: y = X_new_b * theta, shape (2, 1)
print("Normal Equation predictions:", y_predict.ravel())  # Flatten and print, expect ~[4, 10]

# Scikit-Learn Linear Regression (SVD-based)
# Why: Uses SVD for numerical stability, especially when X^T X is not invertible
lin_reg = LinearRegression()  # Initialize LinearRegression model, which uses SVD internally
lin_reg.fit(X, y.ravel())  # Train model on X (shape (100, 1)) and y (flattened to (100,)), handles intercept automatically
print("Scikit-Learn intercept, slope:", lin_reg.intercept_, lin_reg.coef_)  # Print intercept (~4) and slope (~3)
y_predict_sklearn = lin_reg.predict(X_new)  # Predict for x = 0 and x = 2, shape (2,)
print("Scikit-Learn predictions:", y_predict_sklearn)  # Expect ~[4, 10]

# SVD via numpy.linalg.lstsq
# Why: Demonstrates SVD’s robustness for solving least squares problems
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)  # Solve using SVD, rcond cuts off small singular values
# theta_best_svd: Optimal parameters, residuals: Sum of squared errors, rank: Matrix rank, s: Singular values
print("SVD theta (numpy.linalg.lstsq):", theta_best_svd)  # Matches Scikit-Learn, expect ~[4, 3]

# Pseudoinverse via SVD
# Why: Shows how the Moore-Penrose pseudoinverse handles non-invertible matrices
X_b_pinv = np.linalg.pinv(X_b)  # Compute pseudoinverse of X_b using SVD, shape (2, 100)
theta_pinv = X_b_pinv.dot(y)  # Compute theta = X^+ y, shape (2, 1)
print("Pseudoinverse theta:", theta_pinv.ravel())  # Flatten and print, matches SVD results

# Batch Gradient Descent
# Why: Iteratively optimizes theta by following the gradient of the MSE, useful for large datasets
eta = 0.1  # Learning rate, controls step size of updates, set to 0.1 for reasonable convergence
n_iterations = 1000  # Number of iterations to update theta
m = 100  # Number of data points, used to scale gradients
theta = np.random.randn(2, 1)  # Randomly initialize theta (intercept, slope), shape (2, 1)
for iteration in range(n_iterations):  # Loop for 1000 iterations
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)  # Compute gradient of MSE: (2/m) * X^T (X theta - y), shape (2, 1)
    theta = theta - eta * gradients  # Update theta: move opposite to gradient direction
print("Batch Gradient Descent theta:", theta.ravel())  # Flatten and print, expect ~[4, 3]

# Stochastic Gradient Descent (SGD)
# Why: Updates theta using one random sample per iteration, faster but noisier
n_epochs = 50  # Number of epochs, each epoch processes all data points once
t0, t1 = 5, 50  # Learning schedule parameters for decreasing learning rate
def learning_schedule(t):  # Define function to compute learning rate
    return t0 / (t + t1)  # Returns eta(t) = t0 / (t + t1), reduces eta over time
theta = np.random.randn(2, 1)  # Randomly initialize theta, shape (2, 1)
for epoch in range(n_epochs):  # Loop over 50 epochs
    for i in range(m):  # Loop over each data point
        random_index = np.random.randint(m)  # Pick a random index from 0 to 99
        xi = X_b[random_index:random_index+1]  # Select one sample, shape (1, 2)
        yi = y[random_index:random_index+1]  # Select corresponding target, shape (1, 1)
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)  # Compute gradient for one sample, shape (2, 1)
        eta = learning_schedule(epoch * m + i)  # Compute learning rate for current step
        theta = theta - eta * gradients  # Update theta using gradient and learning rate
print("Stochastic Gradient Descent theta:", theta.ravel())  # Flatten and print, expect ~[4, 3]

# Mini-batch Gradient Descent with SGDRegressor
# Why: Balances speed of SGD and stability of Batch GD, optimized for hardware
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)  # Initialize SGDRegressor
# max_iter: Max iterations, tol: Stop if loss improves less than 0.001, penalty=None: No regularization, eta0: Initial learning rate
sgd_reg.fit(X, y.ravel())  # Train on X and flattened y
print("SGDRegressor intercept, slope:", sgd_reg.intercept_, sgd_reg.coef_)  # Print parameters, expect ~[4, 3]

# Section 2: Polynomial Regression
# Purpose: Model a non-linear relationship y = 0.5x^2 + x + 2 + noise
# Why: To show how linear regression can fit non-linear data using polynomial features
np.random.seed(42)  # Reset seed for reproducibility
m = 100  # Number of data points
X = 6 * np.random.rand(m, 1) - 3  # Generate x in [-3, 3], shape (100, 1)
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)  # Compute y = 0.5x^2 + x + 2 + noise, shape (100, 1)

poly_features = PolynomialFeatures(degree=2, include_bias=False)  # Create polynomial features up to degree 2, no intercept
X_poly = poly_features.fit_transform(X)  # Transform X to [x, x^2], shape (100, 2)
lin_reg = LinearRegression()  # Initialize LinearRegression for polynomial features
lin_reg.fit(X_poly, y.ravel())  # Train on polynomial features and flattened y
print("Polynomial Regression intercept, coef:", lin_reg.intercept_, lin_reg.coef_)  # Expect ~2, ~1, ~0.5

# Section 3: Learning Curves
# Purpose: Plot training and validation errors to diagnose model performance
# Why: To identify underfitting (high error) or overfitting (large gap between curves)
def plot_learning_curves(model, X, y):  # Define function to plot learning curves
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data: 80% train, 20% val
    train_errors, val_errors = [], []  # Initialize lists to store MSE for training and validation
    for m in range(1, len(X_train)):  # Loop over training set sizes from 1 to len(X_train)-1
        model.fit(X_train[:m], y_train[:m])  # Train model on first m samples
        y_train_predict = model.predict(X_train[:m])  # Predict on training data
        y_val_predict = model.predict(X_val)  # Predict on validation data
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))  # Compute training MSE
        val_errors.append(mean_squared_error(y_val, y_val_predict))  # Compute validation MSE
    plt.figure(figsize=(8, 6))  # Create new figure with size 8x6 inches
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training")  # Plot training RMSE (red line with + markers)
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation")  # Plot validation RMSE (blue line)
    plt.xlabel("Training set size")  # Label x-axis
    plt.ylabel("RMSE")  # Label y-axis
    plt.title("Learning Curves")  # Set plot title
    plt.legend()  # Show legend
    plt.grid(True)  # Add grid for readability
    plt.show()  # Display the plot

# Plot learning curves for Linear Regression
lin_reg = LinearRegression()  # Initialize LinearRegression model
plot_learning_curves(lin_reg, X, y)  # Plot learning curves for linear model

# Plot learning curves for Polynomial Regression (degree 10)
polynomial_regression = Pipeline([  # Create pipeline to chain polynomial features and regression
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),  # Transform to degree 10 polynomials
    ("lin_reg", LinearRegression())  # Fit linear regression on polynomial features
])
plot_learning_curves(polynomial_regression, X, y)  # Plot learning curves, expect overfitting (low training error, high validation error)

# Section 4: Regularized Models
# Purpose: Use Ridge, Lasso, and Elastic Net to prevent overfitting
# Why: To demonstrate how regularization controls model complexity
ridge_reg = Ridge(alpha=1, solver="cholesky")  # Initialize Ridge (L2 regularization) with alpha=1, using Cholesky solver
ridge_reg.fit(X, y.ravel())  # Train on X and flattened y
print("Ridge prediction for x=1.5:", ridge_reg.predict([[1.5]]))  # Predict for x=1.5

lasso_reg = Lasso(alpha=0.1)  # Initialize Lasso (L1 regularization) with alpha=0.1
lasso_reg.fit(X, y.ravel())  # Train on X and flattened y
print("Lasso prediction for x=1.5:", lasso_reg.predict([[1.5]]))  # Predict for x=1.5

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Initialize Elastic Net (L1+L2) with alpha=0.1, L1 ratio=0.5
elastic_net.fit(X, y.ravel())  # Train on X and flattened y
print("Elastic Net prediction for x=1.5:", elastic_net.predict([[1.5]]))  # Predict for x=1.5

# Early Stopping with SGDRegressor
# Purpose: Stop training when validation error is minimized to avoid overfitting
# Why: To handle high-degree polynomials that overfit without regularization
poly_scaler = Pipeline([  # Create pipeline for polynomial features and scaling
    ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),  # Transform to degree 90 polynomials
    ("std_scaler", StandardScaler())  # Scale features to mean=0, variance=1
])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data
X_train_poly_scaled = poly_scaler.fit_transform(X_train)  # Fit and transform training data
X_val_poly_scaled = poly_scaler.transform(X_val)  # Transform validation data using same parameters
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)  # Initialize SGDRegressor
# max_iter=1: One iteration per fit, warm_start=True: Continue from previous weights, penalty=None: No regularization
minimum_val_error = float("inf")  # Initialize minimum validation error to infinity
best_epoch = None  # Track epoch with lowest validation error
best_model = None  # Store best model
for epoch in range(1000):  # Loop over 1000 epochs
    sgd_reg.fit(X_train_poly_scaled, y_train.ravel())  # Train for one iteration
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)  # Predict on validation set
    val_error = mean_squared_error(y_val, y_val_predict)  # Compute validation MSE
    if val_error < minimum_val_error:  # Check if current error is lowest
        minimum_val_error = val_error  # Update minimum error
        best_epoch = epoch  # Save epoch number
        best_model = clone(sgd_reg)  # Save copy of model
print("Best epoch for early stopping:", best_epoch)  # Print best epoch

# Section 5: Logistic and Softmax Regression
# Purpose: Perform binary and multi-class classification on the Iris dataset
# Why: To demonstrate classification techniques for categorical outcomes
iris = datasets.load_iris()  # Load Iris dataset (150 samples, 4 features)
X = iris["data"][:, 3:]  # Use petal width as feature, shape (150, 1)
y = (iris["target"] == 2).astype(np.int32)  # Binary labels: 1 for Virginica, 0 otherwise, shape (150,)

log_reg = LogisticRegression()  # Initialize LogisticRegression for binary classification
log_reg.fit(X, y)  # Train on petal width and binary labels
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)  # Create 1000 test inputs from 0 to 3, shape (1000, 1)
y_proba = log_reg.predict_proba(X_new)  # Compute probabilities for both classes, shape (1000, 2)
plt.figure(figsize=(8, 6))  # Create new figure
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris Virginica")  # Plot probability of Virginica
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris Virginica")  # Plot probability of not Virginica
plt.xlabel("Petal width (cm)")  # Label x-axis
plt.ylabel("Probability")  # Label y-axis
plt.title("Logistic Regression Probabilities")  # Set title
plt.legend()  # Show legend
plt.grid(True)  # Add grid
plt.show()  # Display plot
print("Logistic Regression predictions for petal width 1.7, 1.5:", log_reg.predict([[1.7], [1.5]]))  # Predict classes

# Softmax Regression (multi-class classification)
X = iris["data"][:, (2, 3)]  # Use petal length and petal width, shape (150, 2)
y = iris["target"]  # Multi-class labels (0, 1, 2 for Setosa, Versicolor, Virginica), shape (150,)
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)  # Initialize for multi-class
# multi_class="multinomial": Use softmax, solver="lbfgs": Optimization algorithm, C=10: Inverse of regularization strength
softmax_reg.fit(X, y)  # Train on two features and multi-class labels
print("Softmax prediction for [5, 2]:", softmax_reg.predict([[5, 2]]))  # Predict class for input [5, 2]
print("Softmax probabilities for [5, 2]:", softmax_reg.predict_proba([[5, 2]]))  # Predict probabilities