# In this file we are going to define functions that will help us find correlation between every kind of features including categorical features and numerical features

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

def non_linear_correlation(column1, column2, k=5):
    '''
    This function will help us find correlation between two columns that are categorical or numerical or one of them is categorical and other is numerical
    :param column1: The column which is supposed to be an independent variable, pandas series
    :param column2: Dependent variable column, pandas series
    :param k: Number of folds for cross-validation
    :return: Return the average score of accuracy or R2 over k splits
    '''
    # Checking if both the columns are categorical
    if column1.dtype == 'object' and column2.dtype == 'object':
        # make a decision tree classifier and fit the column2 on column1
        # return the average accuracy score over k splits
        df = pd.DataFrame({'column1': column1, 'column2': column2})
        df.dropna(inplace=True)
        label_encoder = LabelEncoder()
        df['column1'] = label_encoder.fit_transform(df['column1'])
        df['column2'] = label_encoder.fit_transform(df['column2'])
        df = df.values

        model = DecisionTreeClassifier()

        scores = cross_val_score(model, df[:, 0].reshape(-1, 1), df[:, 1], cv=k, scoring='accuracy')
        avg_score = np.mean(scores)
        avg_score = (avg_score - 1/np.unique(df[:, 1]).shape[0]) / (1 - 1/np.unique(df[:, 1]).shape[0])
        return avg_score

    # Checking if both the columns are numerical
    elif column1.dtype != 'object' and column2.dtype != 'object':
        # make a decision tree regressor and fit the column2 on column1
        # return the average R2 score over k splits
        df = pd.DataFrame({'column1': column1, 'column2': column2})
        df.dropna(inplace=True)
        df = df.values

        model = DecisionTreeRegressor()

        scores = cross_val_score(model, df[:, 0].reshape(-1, 1), df[:, 1], cv=k, scoring='r2')
        avg_score = np.mean(scores)

        return max(avg_score, 0)

    # Checking if one of the columns is categorical and the other is numerical
    elif column1.dtype == 'object' and column2.dtype != 'object':
        # make a decision tree regressor and fit the column2 on column1
        # return the average R2 score over k splits
        df = pd.DataFrame({'column1': column1, 'column2': column2})
        df.dropna(inplace=True)
        label_encoder = LabelEncoder()
        df['column1'] = label_encoder.fit_transform(df['column1'])
        df = df.values

        model = DecisionTreeRegressor()

        scores = cross_val_score(model, df[:, 0].reshape(-1, 1), df[:, 1], cv=k, scoring='r2')
        avg_score = np.mean(scores)

        return max(avg_score, 0)

    # Checking if one of the columns is numerical and the other is categorical
    elif column1.dtype != 'object' and column2.dtype == 'object':
        # make a decision tree classifier and fit the column2 on column1
        # return the average accuracy score over k splits
        df = pd.DataFrame({'column1': column1, 'column2': column2})
        df.dropna(inplace=True)
        label_encoder = LabelEncoder()
        df['column2'] = label_encoder.fit_transform(df['column2'])
        df = df.values

        model = DecisionTreeClassifier()

        scores = cross_val_score(model, df[:, 0].reshape(-1, 1), df[:, 1], cv=k, scoring='accuracy')
        avg_score = np.mean(scores)
        avg_score = (avg_score - 1/np.unique(df[:, 1]).shape[0]) / (1 - 1/np.unique(df[:, 1]).shape[0])
        return avg_score

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
            #print('The correlation between {} and {} is {}'.format(i, j, correlation_df.loc[i, j]))
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
    #print('Correlation dataframe is ready')
    # plot the heatmap
    sns.heatmap(correlation_df, annot=True)
    # add the dependent and independent variable label , y axis - independent variable, x axis - dependent variable , title - non linear correlation heatmap
    plt.ylabel('Independent Variable')
    plt.xlabel('Dependent Variable')
    plt.title('Non Linear Correlation Heatmap')
    plt.show()