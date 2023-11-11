# In this file we are going to define functions that will help us find correlation between every kind of features including categorical features and numerical features

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder

def non_linear_correlation(column1, column2):
    '''
    This function will help us find correlation between two columns that are categorical or numerical or one of them is categorical and other is numerical
    :param column1: The column which is supposed to be a independent variable, pandas series
    :param column2: Dependent variable column, pandas series
    :return: Return the score of accuracy of production between these columns , by keeping column one as independent and column 2 as dependent variable
    '''
    # Checking if both the columns are categorical
    if column1.dtype == 'object' and column2.dtype == 'object':
        # make a decision tree classifier and fit the column2 on column1
        # return the accuracy score
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        # make a pandas dataframe of the two columns
        df = pd.DataFrame()
        df['column1'] = column1
        df['column2'] = column2
        # drop the rows where there are null values
        df.dropna(inplace=True)
        # Encoding the categorical columns
        label_encoder = LabelEncoder()
        df['column1'] = label_encoder.fit_transform(df['column1'])
        df['column2'] = label_encoder.fit_transform(df['column2'])
        # Splitting the data into train and test
        df = df.values
        x_train, x_test, y_train, y_test = train_test_split(df[:, 0], df[:, 1], test_size=0.2, random_state=0)
        # Making the decision tree classifier
        model = DecisionTreeClassifier()
        # Fitting the model
        model.fit(x_train.reshape(-1, 1), y_train)
        # Predicting the values
        y_pred = model.predict(x_test.reshape(-1, 1))
        # Finding the accuracy score
        score = accuracy_score(y_test, y_pred)
        return score

    # Checking if both the columns are numerical
    elif column1.dtype != 'object' and column2.dtype != 'object':
        # make a decision tree regressor and fit the column2 on column1
        # return the accuracy score
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        # make a pandas dataframe of the two columns
        df = pd.DataFrame()
        df['column1'] = column1
        df['column2'] = column2
        # drop the rows where there are null values
        df.dropna(inplace=True)
        # Splitting the data into train and test
        df = df.values
        x_train, x_test, y_train, y_test = train_test_split(df[:,0], df[:,1], test_size=0.2, random_state=0)
        # Making the decision tree regressor
        model = DecisionTreeRegressor()
        # Fitting the model
        model.fit(x_train.reshape(-1, 1), y_train)
        # Predicting the values
        y_pred = model.predict(x_test.reshape(-1, 1))
        # Finding the accuracy score
        score = r2_score(y_test, y_pred)
        return score

    # Checking if one of the columns is categorical and other is numerical, column1 is categorical and column2 is numerical
    elif column1.dtype == 'object' and column2.dtype != 'object':
        # make a decision tree regressor and fit the column2 on column1
        # return the accuracy score
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        # make a pandas dataframe of the two columns
        df = pd.DataFrame()
        df['column1'] = column1
        df['column2'] = column2
        # drop the rows where there are null values
        df.dropna(inplace=True)
        # Encoding the categorical columns
        label_encoder = LabelEncoder()
        df['column1'] = label_encoder.fit_transform(df['column1'])
        # Splitting the data into train and test
        df = df.values
        x_train, x_test, y_train, y_test = train_test_split(df[:, 0], df[:, 1], test_size=0.2, random_state=0)
        # Making the decision tree regressor
        model = DecisionTreeRegressor()
        # Fitting the model
        model.fit(x_train.reshape(-1, 1), y_train)
        # Predicting the values
        y_pred = model.predict(x_test.reshape(-1, 1))
        # Finding the accuracy score
        score = r2_score(y_test, y_pred)
        return score

    # Checking if one of the columns is categorical and other is numerical, column1 is numerical and column2 is categorical
    elif column1.dtype != 'object' and column2.dtype == 'object':
        # make a decision tree regressor and fit the column2 on column1
        # return the accuracy score
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score

        # make a pandas dataframe of the two columns
        df = pd.DataFrame()
        df['column1'] = column1
        df['column2'] = column2
        # drop the rows where there are null values
        df.dropna(inplace=True)
        # Encoding the categorical columns
        label_encoder = LabelEncoder()
        df['column2'] = label_encoder.fit_transform(df['column2'])
        # Splitting the data into train and test
        df = df.values
        x_train, x_test, y_train, y_test = train_test_split(df[:, 0], df[:, 1], test_size=0.2, random_state=0)
        # Making the decision tree regressor
        model = DecisionTreeClassifier()
        # Fitting the model
        model.fit(x_train.reshape(-1,1), y_train)
        # Predicting the values
        y_pred = model.predict(x_test.reshape(-1, 1))
        # Finding the accuracy score
        score = r2_score(y_test, y_pred)
        return score

# make a function that will take a pandas dataframe and return a pandas dataframe with all the correlation values based on the above function
def correlation_dataframe(df):
    '''
    This function will take a pandas dataframe and return a pandas dataframe with all the correlation values based on the above function
    :param df: pandas dataframe
    :return: pandas dataframe with correlation values
    '''
    # make a empty dataframe
    correlation_df = pd.DataFrame()
    # iterate through the columns of the dataframe
    for i in df.columns:
        for j in df.columns:
            # find the correlation between the two columns
            correlation_df.loc[i, j] = non_linear_correlation(df[i], df[j])
            # print the correlation value
            print('The correlation between {} and {} is {}'.format(i, j, correlation_df.loc[i, j]))
    return correlation_df

# make a function that will take a Pandas data frame and draw or display a heat map of correlation values between the columns
def correlation_heatmap(df):
    '''
    This function will take a Pandas data frame and draw or display a heat map of correlation values between the columns
    :param df: pandas dataframe
    :return: None
    '''
    # make a empty dataframe
    correlation_df = correlation_dataframe(df)
    # print correlation datafram is ready
    print('Correlation dataframe is ready')
    # plot the heatmap
    sns.heatmap(correlation_df, annot=True)
    plt.show()