# import necessary libraries
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np


def count_nans(data):
    """
    Function that counts number None cells in df
    """
    return pd.isna(data).sum().sum()


def draw_scatter_data(x, y, xlabel_name, ylabel_name):
    """
    Function that plot the given data
    """
    legend = plt.scatter(x, y, s=4, c='crimson', label = "data points")
    plt.legend(handles = [legend])
    plt.title(ylabel_name + " / " + xlabel_name)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.show()


def plot_data(x, y, xlabel_name, ylabel_name):
    """
    Function that plot the given data
    """
    plt.plot(x, y, c="crimson")
    plt.title(ylabel_name + " / " + xlabel_name)
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.show()


def splitting_data_by_year(Data):
    """
    Function that split the data by 2018 year
    """
    # split our data to test and train with condition: year for train < 2018
    train_data = Data[pd.to_datetime(Data['Scheduled depature time']).dt.year < 2018]
    test_data = Data[pd.to_datetime(Data['Scheduled depature time']).dt.year == 2018]
    # Drop useless features
    train_data = train_data.drop(['Scheduled depature time', 'Scheduled arrival time'], axis=1)
    test_data = test_data.drop(['Scheduled depature time', 'Scheduled arrival time'], axis=1)
    return train_data, test_data


def split_to_data_and_labels(train_data, test_data):
    """
    Function that split the data to X and Y
    """
    #get labels (Delay) from train and test data
    train_labels = train_data['Delay']
    test_labels = test_data['Delay']
    # remove labels (Delay) from data because we do not need it
    train_data = train_data.drop('Delay', axis=1)
    test_data = test_data.drop('Delay', axis=1)
    return train_data, train_labels, test_data, test_labels


def outlier_removal(x, threshold):
    """
    Function that removes outliers from given data
    """
    # compute z-score in all data
    z = np.abs(stats.zscore(x))
    # remove data which z-score less then threshold
    x = x[(z < threshold).all(axis = 1)]
    return x


def data_preparation(path_to_data):
    """
    Main function for data processing
    """
    print("{:|^80s}".format("> Start file preparing <"))
    # read all the data from .csv file
    Data = pd.read_csv("./flight_delay.csv")
    # Now we need to understand the data. Let's see how much unique features we have.
    # Print count and types of categorical feature's
    types = Data.dtypes
    print("{:-^80s}".format(" default features "))
    print("Number categorical features: ", sum(types == 'object'))
    print(types)
    # Before splitting the data we need to ensure that our data have not a "None" values
    # Otherwise we have to do imputing
    print("{:-^80s}".format(" None values check "))
    print("number of nan in data: ", count_nans(Data))
    if count_nans(Data) > 0:
        raise ValueError('Our data has a Nan values! Please, use an imputer :)')
    # Prepare label encoder, because we can not put not a number data to regression model
    label_encoder = preprocessing.LabelEncoder()
    # Get columns with time from data
    departure_airport_feature = Data["Depature Airport"]
    destination_airport_feature = Data["Destination Airport"]
    # Append lists for mapping
    merge_airport_feature = destination_airport_feature.append(departure_airport_feature)
    # Load list to label encoder
    label_encoder.fit(merge_airport_feature)
    # Update values
    Data["Depature Airport"] = label_encoder.transform(departure_airport_feature)
    Data["Destination Airport"] = label_encoder.transform(destination_airport_feature)
    # Prepare time features from other columns
    #   Convert object to datetime type
    departure_time_feature = pd.to_datetime(Data["Scheduled depature time"])
    arrival_time_feature = pd.to_datetime(Data["Scheduled arrival time"])
    # Get the flight time
    flight_time = arrival_time_feature - departure_time_feature
    # Add new features to Data
    Data["Depature month"] = departure_time_feature.dt.month
    Data["Arrival month"] = arrival_time_feature.dt.month
    Data["Depature hour"] = departure_time_feature.dt.hour
    Data["Arrival hour"] = arrival_time_feature.dt.hour
    # Convert flight time to minutes
    Data["flight time"] = flight_time.dt.total_seconds() / 60
    # Split data by year
    train, test = splitting_data_by_year(Data)
    # Remove outliers from all data
    draw_scatter_data(train["flight time"], train["Delay"], "Flight time", "Delay")
    train = outlier_removal(train, 3)
    # Finally split data without outliers to X and Y
    x_train, y_train, x_test, y_test = split_to_data_and_labels(train, test)
    print("{:|^80s}".format("> End file preparing <"))
    return x_train, y_train, x_test, y_test
