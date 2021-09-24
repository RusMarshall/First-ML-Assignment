# import necessary libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plt
from sklearn import metrics
import Data_operations
import numpy as np


def multiple_linear_regression(x_train, y_train, x_test, y_test):
    """
    Function which apply multiple linear regression model
    """
    print("{:=^80s}".format("> Multiple linear regression <"))
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    print(f"Model intercept : {regressor.intercept_}")
    print(f"Model coefficients : {regressor.coef_}")
    y_pred = regressor.predict(x_test)
    calculate_metrics(y_test, y_pred)
    print("{:=^80s}".format("> Train results <"))
    y_pred_train = regressor.predict(x_train)
    calculate_metrics(y_train, y_pred_train)
    plot_several_scatters(x_train, y_train, x_test, y_test, y_pred)


def polyminal_regression(x_train, y_train, x_test, y_test):
    """
    Function which apply polynomial regression model
    """
    print("{:=^80s}".format("> Polyminal regression <"))
    degrees = [1, 4, 9]
    X = x_train["flight time"]
    XT = x_test["flight time"]
    for i in range(len(degrees)):
        polynomial_features = PolynomialFeatures(degree=degrees[i])
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(X[:, np.newaxis], y_train)

        # Evaluate the models using crossvalidation
        scores = cross_val_score(pipeline, XT[:, np.newaxis], y_test,
                                 scoring="neg_mean_squared_error", cv=10)
        y_pred = pipeline.predict(XT[:, np.newaxis])
        print(f"degree {degrees[i]}")
        calculate_metrics(y_test, y_pred)
        print("{:=^80s}".format("> Train results <"))
        y_pred_train = pipeline.predict(X[:, np.newaxis])
        print(f"degree {degrees[i]}")
        calculate_metrics(y_train, y_pred_train)
        plot_several_scatters(x_train, y_train, x_test, y_test, y_pred)


def lasso_regressor(x_train, y_train, x_test, y_test):
    """
    Function which apply lasso regressor model
    """
    print("{:=^80s}".format("> Lasso regressor <"))
    alphas = [6, 4.5, 4, 3, 2, 1.5, 1.4, 1.3, 1.2, 1, 0.5, 0.1]
    losses = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha)
        lasso.fit(x_train, y_train)
        prediction = lasso.predict(x_test)
        MSE = mean_squared_error(y_test, prediction)
        losses.append(MSE)

    Data_operations.plot_data(alphas, losses, "alpha", "Mean squared error")
    best_alpha = alphas[np.argmin(losses)]
    print("Best value of alpha:", best_alpha)
    lasso = Lasso(best_alpha)
    lasso.fit(x_train, y_train)
    y_pred = lasso.predict(x_test)
    calculate_metrics(y_test, y_pred)
    print("{:=^80s}".format("> Train results <"))
    y_pred_train = lasso.predict(x_train)
    calculate_metrics(y_train, y_pred_train)
    plot_several_scatters(x_train, y_train, x_test, y_test, y_pred)


def calculate_metrics(y_test, y_pred):
    """
    Function which computes three metrics:
    •	Mean Squared Error (MSE)
    •	Root Mean Squared Error (RMSE)
    •	Mean Absolute Error (MAE)
    """
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.r2_score(y_test, y_pred)))


def plot_several_scatters(x_train, y_train, x_test, y_test, y_pred):
    """
    Function which plots three data sets:
    •	test data
    •	train data
    •	predicted test data
    """
    test_data = plt.scatter(x_test["flight time"], y_test, s=4, c='crimson', label="test data")
    train_data = plt.scatter(x_train["flight time"], y_train, s=4, c='maroon', label="train data")
    predicted_data = plt.scatter(x_test["flight time"], y_pred, s=4, c='cyan', label="predicted data")
    plt.title("lasso regressor")
    plt.legend(handles=[test_data, train_data, predicted_data])
    plt.show()