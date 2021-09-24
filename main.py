# import necessary libraries
import Data_operations
import Algorithms

if __name__ == '__main__':
    # Preparation our data to train and test with extra features like label encoding, outlier removal etc.
    x_train, y_train, x_test, y_test = Data_operations.data_preparation("./flight_delay.csv")
    print(x_train.head().to_string())
    # Plot data
    Data_operations.draw_scatter_data(x_train["flight time"], y_train, "Flight time", "Delay")
    Data_operations.draw_scatter_data(x_train["Depature Airport"], y_train, "Depature Airport", "Delay")
    Data_operations.draw_scatter_data(x_train["Destination Airport"], y_train, "Destination Airport", "Delay")
    Data_operations.draw_scatter_data(x_train["Arrival month"], y_train, "Scheduled arrival month", "Delay")
    Data_operations.draw_scatter_data(x_train["Depature month"], y_train, "Scheduled depature month", "Delay")
    Data_operations.draw_scatter_data(x_train["Depature hour"], y_train, "Scheduled depature hour", "Delay")
    Data_operations.draw_scatter_data(x_train["Arrival hour"], y_train, "Scheduled arrival hour", "Delay")
    # Fit data to the ML algorithms
    Algorithms.multiple_linear_regression(x_train, y_train, x_test, y_test)
    Algorithms.polyminal_regression(x_train, y_train, x_test, y_test)
    Algorithms.lasso_regressor(x_train, y_train, x_test, y_test)