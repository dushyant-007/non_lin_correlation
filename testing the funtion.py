from non_linear_correlation import non_linear_correlation
from non_linear_correlation import correlation_heatmap
import pandas as pd
import numpy as np
import seaborn as sns


# start the main function
if __name__ == "__mai1n__":
    # make two columns of numerical data that is slightly correlated
    column1 = np.random.randint(1, 100, 100)
    column2 = column1 + 0.25 * np.random.randint(1, 10, 100) * np.random.randint(1, 10, 100)
    # Convert to pandas Series with float data type
    column1, column2 = pd.Series(column1, dtype=float), pd.Series(column2, dtype=float)
    # Introduce missing values to the columns
    column1[5:8] = np.nan
    column2[15:20] = np.nan
    print(non_linear_correlation(column1, column2))
    # make two columns of categorical data that is slightly correlated
    column1 = np.random.randint(1, 100, 100)
    column2 = ['apple' if (i < 25 or i>74) else 'orange' for i in column1]
    column2 = pd.Series(column2)
    # find the correlation between these two columns
    column1, column2 = pd.Series(column1), pd.Series(column2)
    print(non_linear_correlation(column1, column2))

    # make one column of categorical data and other of numerical data that is slightly correlated
    column1 = np.random.randint(1, 100, 100)
    column1 = ['apple' if (i < 25 or i>74) else 'orange' for i in column1]
    column1 = pd.Series(column1)
    # make column2 such that if column1 is apple than a random number is generated between 1 and 25 and if column1 is orange than a random number is generated between 75 and 100
    column2 = [np.random.randint(1, 25) if i == 'apple' else np.random.randint(75, 100) for i in column1]
    column2 = pd.Series(column2)
    # find the correlation between these two columns
    column1[5:8] = [None] * len(column1[5:8])
    column2[15:20] = [None] * len(column2[15:20])
    column1, column2 = pd.Series(column1), pd.Series(column2)
    print(non_linear_correlation(column1, column2))

    # make two catergorical columns that are correlated with each other
    column1 = np.random.randint(1, 100, 100)
    column1 = ['apple' if (i < 50 and i>25) else 'orange' for i in column1]
    # make column2 such that if column1 is apple than a random number is column2 could be either 3,4 and if column1 is orange than a random number is generated between 8,9
    column2 = [np.random.randint(3, 5) if i == 'apple' else np.random.randint(8, 10) for i in column1]
    # put some nulls in column1 and column2
    column1[5:8] = [None] * len(column1[5:8])
    column2[15:20] = [None] * len(column2[15:20])
    # find the correlation between these two columns
    column1, column2 = pd.Series(column1), pd.Series(column2)
    print(non_linear_correlation(column1, column2))


if __name__ == "__main__":
    # Load Titanic dataset
    titanic = sns.load_dataset('titanic')

    # Display the first few rows of the dataset
    print(titanic.info())
    print(titanic.head())

    # Select numerical and categorical columns for demonstration
    numerical_cols = ['age', 'fare', 'pclass']
    categorical_cols = ['sex', 'embarked', 'who']

    # the code is based on the dtype , so if the dtype if none of int, float, bool, then it must be 'object' nothing else.
    # Combine selected columns into a new DataFrame
    selected_columns = numerical_cols + categorical_cols
    titanic_df = titanic[selected_columns]

    # Display the first few rows of the new dataset
    print(titanic_df.info())
    print(titanic_df.head())

    # Make the correlation heatmap
    correlation_heatmap(titanic_df)



